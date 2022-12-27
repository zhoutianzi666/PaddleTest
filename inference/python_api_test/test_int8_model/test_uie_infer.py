# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import json
import sys
from functools import partial
import distutils.util
import numpy as np

import paddle
from paddle import inference
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.metrics import SpanEvaluator
from backend import PaddleInferenceEngine, TensorRTEngine, ONNXRuntimeEngine, Monitor

def argsparser():
    """
    parse_args func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./afqmc",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="inference.pdmodel",
        help="model file name")
    parser.add_argument(
        "--params_filename",
        type=str,
        default="inference.pdiparams",
        help="params file name")
    parser.add_argument(
        "--dev_data",
        default="./data/dev.txt",
        type=str,
        help="The data file of validation.", )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.", )
    parser.add_argument(
        "--use_trt",
        action="store_true",
        help="Whether to use inference engin TensorRT.", )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'.",
    )
    parser.add_argument(
        "--use_mkldnn",
        type=bool,
        default=False,
        help="Whether use mkldnn or not.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of cpu threads.")

    parser.add_argument(
        "--deploy_backend",
        type=str,
        default="paddle_inference",
        help="deploy backend, it can be: `paddle_inference`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
    parser.add_argument("--calibration_file", type=str, default=None, help="quant onnx model calibration cache file.")
    return parser


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def _convert_example(example, tokenizer, max_seq_length=128):
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_seq_len=max_seq_length,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=False,
        return_offsets_mapping=True)
    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0.0 for x in range(max_seq_length)]
    end_ids = [0.0 for x in range(max_seq_length)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0
    tokenized_output = {
        "input_ids": encoded_inputs["input_ids"],
        "token_type_ids": encoded_inputs["token_type_ids"],
        "start_ids": start_ids,
        "end_ids": end_ids
    }
    return tokenized_output

class WrapperPredictor(object):
    """
    Inference Predictor class
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def predict_batch(self, data):
        """
        predict from batch func
        """
        self.predictor.prepare_data(data)
        output = self.predictor.run()
        return output

    def _convert_predict_batch(self, FLAGS, data, tokenizer, batchify_fn):
        examples = []
        for example in data:
            example = _convert_example(
                example, tokenizer, max_seq_length=FLAGS.max_seq_length)
            examples.append(example)

        return examples

    def eval(self, dataset, tokenizer, batchify_fn, FLAGS):
        """
        predict func
        """
        batches = [
            dataset[idx:idx + FLAGS.batch_size]
            for idx in range(0, len(dataset), FLAGS.batch_size)
        ]

        for i, batch in enumerate(batches):
            examples = self._convert_predict_batch(FLAGS, batch, tokenizer,
                                                   batchify_fn)
            input_ids, segment_ids, start_ids, end_ids = batchify_fn(examples)
            output = self.predict_batch([input_ids, segment_ids])
            if i > FLAGS.perf_warmup_steps:
                break

        metric = SpanEvaluator()
        metric.reset()
        predict_time = 0.0
        for i, batch in enumerate(batches):
            examples = self._convert_predict_batch(FLAGS, batch, tokenizer,
                                                   batchify_fn)
            input_ids, segment_ids, start_ids, end_ids = batchify_fn(examples)
            start_time = time.time()
            output = self.predict_batch([input_ids, segment_ids])
            end_time = time.time()
            predict_time += end_time - start_time
            start_ids = paddle.to_tensor(np.array(start_ids))
            end_ids = paddle.to_tensor(np.array(end_ids))

            start_prob = paddle.to_tensor(output[0])
            end_prob = paddle.to_tensor(output[1])
            num_correct, num_infer, num_label = metric.compute(
                start_prob, end_prob, start_ids, end_ids)
            metric.update(num_correct, num_infer, num_label)

        sequences_num = i * FLAGS.batch_size
        print(
            "[benchmark]batch size: {} Inference time per batch: {}ms, qps: {}.".
            format(
                FLAGS.batch_size,
                round(predict_time * 1000 / i, 2),
                round(sequences_num / predict_time, 2), ))
        precision, recall, f1 = metric.accumulate()
        print("[benchmark]f1: %s. \n" % (f1), end="")
        sys.stdout.flush()


def reader_proprecess(data_path, max_seq_len=128):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content'].strip()
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError(
                    "The value of max_seq_len is too small, please set a larger value"
                )
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result[
                                'end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [
                                    result for result in result_list
                                ]
                                break
                        else:
                            break

                    json_line = {
                        'content': cur_content,
                        'result_list': cur_result_list,
                        'prompt': prompt
                    }
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {
                            'content': res_content,
                            'result_list': result_list,
                            'prompt': prompt
                        }
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def main(FLAGS):
    """
    main func
    """
    paddle.seed(42)
    if FLAGS.use_mkldnn:
        paddle.set_device("cpu")

    token_dir = FLAGS.model_path
    predictor = None
    if FLAGS.deploy_backend == "paddle_inference":
        predictor = PaddleInferenceEngine(
            model_dir=FLAGS.model_path,
            model_filename=FLAGS.model_filename,
            params_filename=FLAGS.params_filename,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            batch_size=FLAGS.batch_size,
            device=FLAGS.device,
            min_subgraph_size=3,
            use_dynamic_shape=FLAGS.use_dynamic_shape,
            cpu_threads=FLAGS.cpu_threads,
        )
    elif FLAGS.deploy_backend == "tensorrt":
        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        token_dir = os.path.dirname(FLAGS.model_path)
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        predictor = TensorRTEngine(
            onnx_model_file=FLAGS.model_path,
            shape_info={
                "input_ids": [[20, 128], [32, 128], [32, 128]],
                "token_type_ids": [[20, 128], [32, 128], [32, 128]],
            },
            max_batch_size=FLAGS.batch_size,
            precision=FLAGS.precision,
            engine_file_path=engine_file,
            calibration_cache_file=FLAGS.calibration_file,
            verbose=False,
        )
    elif FLAGS.deploy_backend == "onnxruntime":
        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        token_dir = os.path.dirname(FLAGS.model_path)
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        predictor = ONNXRuntimeEngine(
            onnx_model_file=FLAGS.model_path,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            device=FLAGS.device,
        )

    dev_ds = load_dataset(
        reader_proprecess, data_path=FLAGS.dev_data, lazy=False)
    tokenizer = AutoTokenizer.from_pretrained(token_dir)
    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        'start_ids': Stack(dtype="int64"),
        'end_ids': Stack(dtype="int64")}): fn(samples)

    WrapperPredictor(predictor).eval(dev_ds, tokenizer, batchify_fn, FLAGS)

    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False
    if rerun_flag:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")


if __name__ == "__main__":
    paddle.set_device("cpu")
    parser = argsparser()
    FLAGS = parser.parse_args()
    main(FLAGS)
