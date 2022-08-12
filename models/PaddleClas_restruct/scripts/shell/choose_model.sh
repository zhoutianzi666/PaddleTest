echo $input_model_type
echo "######  pdparams_pretrain"
echo $pdparams_pretrain
if [[ ${predict_step} == "" ]];then     #要区分下不能把之前的训好的覆盖了
    if [[ ${input_model_type} == "trained" ]];then
        if [[ -f ${output_dir}/${model_name}/${pdparams_pretrain}/latest.pdparams ]];then
            export pretrained_model=${output_dir}/${model_name}/${pdparams_pretrain}/latest
        else
            export pretrained_model="None"  #使用初始化参数评估
        fi
    elif [[ ${input_model_type} == "pretrained" ]];then
        # PaddleClas/ppcls/arch/backbone/legendary_models/
        # esnet.py    ESNet
        # hrnet.py    HRNet
        # inception_v3.py     InceptionV3
        # mobilenet_v1.py     MobileNetV1
        # mobilenet_v3.py     MobileNetV3
        # pp_hgnet.py         PPHGNet
        # pp_lcnet.py         PPLCNet
        # pp_lcnet_v2.py      PPLCNetV2
        # resnet.py       ResNet
        # swin_transformer.py     SwinTransformer
        # vgg.py      VGG
        if [[ -f ${pdparams_pretrain}_pretrained.pdparams ]];then #有下载好的跳过下载
            export pretrained_model=${pdparams_pretrain}_pretrained
        else
            if [[ ${pdparams_pretrain} =~ "ESNet" ]] || [[ ${pdparams_pretrain} =~ "HRNet" ]] || [[ ${pdparams_pretrain} =~ "InceptionV3" ]] || \
                [[ ${pdparams_pretrain} =~ "MobileNetV1" ]] || [[ ${pdparams_pretrain} =~ "MobileNetV3" ]] || [[ ${pdparams_pretrain} =~ "PPHGNet" ]] || \
                [[ ${pdparams_pretrain} =~ "PPLCNet" ]] || [[ ${pdparams_pretrain} =~ "PPLCNetV2" ]] || [[ ${pdparams_pretrain} =~ "ResNet" ]] || \
                [[ ${pdparams_pretrain} =~ "SwinTransformer" ]] || [[ ${pdparams_pretrain} =~ "VGG" ]];then
                echo "######  use legendary_models pretrain model"
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/${pdparams_pretrain}_pretrained.pdparams --no-proxy
            else
                wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/${pdparams_pretrain}_pretrained.pdparams --no-proxy
            fi
            if [[ $? -eq 0 ]];then
                export pretrained_model=${pdparams_pretrain}_pretrained
            else
                echo "\033[31m failed! eval pretrained download ${model_name}/${pdparams_pretrain} failed!\033[0m"
                export pretrained_model=${output_dir}/${model_name}/${pdparams_pretrain}/latest
            fi
        fi
        # 单独考虑
        # if [[ ${model} =~ 'distill_pphgnet_base' ]]  || [[ ${model} =~ 'PPHGNet_base' ]] ;then
        #     echo "######  use distill_pphgnet_base pretrain model"
        #     echo ${model}
        #     echo ${pdparams_pretrain}
        #     wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPHGNet_base_ssld_pretrained.pdparams --no-proxy
        #     rm -rf output/$pdparams_pretrain/latest.pdparams
        #     \cp -r -f PPHGNet_base_ssld_pretrained.pdparams output/$pdparams_pretrain/latest.pdparams
        #     rm -rf PPHGNet_base_ssld_pretrained_pretrained.pdparams
        # fi

        # if [[ ${model} =~ 'PPLCNet' ]]  && [[ ${model} =~ 'dml' ]] ;then #注意区分dml 与 udml
        #     echo "######  use PPLCNet dml pretrain model"
        #     echo ${model}
        #     echo ${pdparams_pretrain}
        #     wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Distillation/${model}_pretrained.pdparams --no-proxy
        #     rm -rf output/$pdparams_pretrain/latest.pdparams
        #     \cp -r -f ${model}_pretrained.pdparams output/$pdparams_pretrain/latest.pdparams
        #     rm -rf ${model}_pretrained.pdparams
        # fi

    else
        export pretrained_model="None"  #使用初始化参数评估
    fi
else
    if [[ ${input_model_type} == "trained" ]];then
        if [[ -f "inference/${model_name}/inference.pdmodel" ]];then
            export pretrained_model="../inference/${model_name}"
        else
            export pretrained_model="None" #必须有下好的模型，不能使用初始化模型，所以不管用默认参数还是None都不能预测
        fi
    elif [[ ${input_model_type} == "pretrained" ]];then
        if [[ -d ${infer_pretrain}_infer ]] && [[ -f ${infer_pretrain}_infer/inference.pdiparams ]];then #有下载好的，或者export已导出的跳过下载
            export pretrained_model="../${infer_pretrain}_infer"
        elif [[ ${model_type} == "PULC" ]];then
            wget -q https://paddleclas.bj.bcebos.com/models/PULC/${infer_pretrain}_infer.tar --no-proxy
        else
            wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/${infer_pretrain}_infer.tar --no-proxy
        fi
        if [[ $? -eq 0 ]];then
            tar xf ${infer_pretrain}_infer.tar
            export pretrained_model="../${infer_pretrain}_infer"
        else
            echo "\033[31m failed! predict pretrained download ${model_name}/${infer_pretrain} failed!\033[0m"
            export pretrained_model="../inference/${model_name}"
        fi
    else
        export pretrained_model="None"
    fi
fi
echo ${pretrained_model}
