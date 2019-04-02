from tools.pytorch_to_lightnet import PytorchToLightNet as C


def writeConvDW(convdw, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        C.writeConvRelu6(convdw._modules['0'], name + "_conv0", parent_name, lntxt, lnweights)
        return C.writeConvRelu6(convdw._modules['2'], name + "_conv1", "previous", lntxt, lnweights)
    else:
        C.writeConv(convdw._modules['0'], name + "_conv0", parent_name, lntxt, lnweights)
        C.writeReLU6(convdw._modules['1'], name + "_relu0", "previous", lntxt, lnweights)

        C.writeConv(convdw._modules['2'], name + "_conv1", "previous", lntxt, lnweights)
        return C.writeReLU6(convdw._modules['3'], name + "_relu1", "previous", lntxt, lnweights)


def writeConvFCN(convfcn, name, parent_name, lntxt, lnweights, useConvRelu):
    writeConvDW(convfcn._modules['0'].conv, name + "_dw0", parent_name, lntxt, lnweights, useConvRelu)
    writeConvDW(convfcn._modules['1'].conv, name + "_dw1", parent_name, lntxt, lnweights, useConvRelu)
    writeConvDW(convfcn._modules['2'].conv, name + "_dw2", parent_name, lntxt, lnweights, useConvRelu)
    writeConvDW(convfcn._modules['3'].conv, name + "_dw3", parent_name, lntxt, lnweights, useConvRelu)

    return name + "_fcn3"


def writeConvBN(convbn, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        return C.writeConvRelu6(convbn._modules['0'], name + "_conv0", parent_name, lntxt, lnweights)
    else:
        C.writeConv(convbn._modules['0'], name + "_conv0", parent_name, lntxt, lnweights)
        return C.writeReLU6(convbn._modules['1'], name + "_relu0", "previous", lntxt, lnweights)


def writeInvertedResidual(ir, use_res_connect, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        C.writeConvRelu6(ir._modules['0'], name + "_conv0", parent_name, lntxt, lnweights)
        C.writeConvRelu6(ir._modules['2'], name + "_conv1", "previous", lntxt, lnweights)
    else:
        C.writeConv(ir._modules['0'], name + "_conv0", parent_name, lntxt, lnweights)
        C.writeReLU6(ir._modules['1'], name + "_relu0", "previous", lntxt, lnweights)

        C.writeConv(ir._modules['2'], name + "_conv1", "previous", lntxt, lnweights)
        C.writeReLU6(ir._modules['3'], name + "_relu1", "previous", lntxt, lnweights)

    C.writeConv(ir._modules['4'], name + "_conv2", "previous", lntxt, lnweights)

    if use_res_connect:
        return C.writeSum(name + "_sum", parent_name, name + "_conv2", lntxt)

    return name + "_conv2"


def writeRPN_Conv(rpn_conv, name, parent_name, lntxt, lnweights, useConvRelu):
    if useConvRelu:
        return C.writeConvRelu6(rpn_conv + "_conv", name, parent_name, lntxt, lnweights)
    else:
        C.writeConv(rpn_conv, name + "_conv", parent_name, lntxt, lnweights)
        return C.writeReLU("stub", name + "_relu", "previous", lntxt, lnweights)


def writeRPN_ClsScore(cls_score, name, parent_name, lntxt, lnweights):
    return C.writeConv(cls_score, name, parent_name, lntxt, lnweights)


def writeRPN_BboxPred(bbox_pred, name, parent_name, lntxt, lnweights):
    return C.writeConv(bbox_pred, name, parent_name, lntxt, lnweights)


def writeLH_FeatureTransform(lh, name, parent_name, lntxt, lnweights):
    C.writeConv(lh.col_conv._modules['0'], name + "_sc_col_conv0", parent_name, lntxt, lnweights)
    col_conv = C.writeConv(lh.col_conv._modules['1'], name + "_sc_col_conv1", "previous", lntxt, lnweights)

    C.writeConv(lh.row_conv._modules['0'], name + "_sc_row_conv0", parent_name, lntxt, lnweights)
    row_conv = C.writeConv(lh.row_conv._modules['1'], name + "_sc_row_conv1", "previous", lntxt, lnweights)

    sum_conv = C.writeSum(name + "_sc_conv_sum", col_conv, row_conv, lntxt)

    return C.writeReLU("", name + "_sc_relu", sum_conv, lntxt, lnweights)


def writeUpscale(upscale, name, parent_name, lntxt, lnweights):
    # upscale_dw = copy.deepcopy(upscale.upconv)
    # upscale_pw = copy.deepcopy(upscale.upconv_pw)
    #
    # bias_dw = copy.deepcopy(upscale_dw.bias.data.cpu().numpy())
    # w_pw = copy.deepcopy(upscale_pw.weight.data.cpu().numpy())
    # bias_pw = copy.deepcopy(upscale_pw.bias.data.cpu().numpy())
    #
    # bias_dw = np.squeeze(bias_dw)
    # bias_dw = bias_dw[:, np.newaxis]
    # w_pw = np.squeeze(w_pw)
    # bias_pw = np.squeeze(bias_pw)
    # bias_pw = bias_pw[:, np.newaxis]
    #
    # new_bias_cw = np.matmul(w_pw, bias_dw) + bias_pw
    #
    # upscale_dw.bias = None
    # upscale_pw.bias.data = torch.tensor(new_bias_cw)

    C.writeConvTransposed(upscale.upconv, name + "_dw", parent_name, lntxt, lnweights)
    C.writeReLU("", name + "_dw_relu", "previous", lntxt, lnweights)

    C.writeConv(upscale.upconv_pw, name + "_pw", "previous", lntxt, lnweights)
    return C.writeReLU("", name + "_relu", "previous", lntxt, lnweights)


def writeMobileRCNN_base(model, lntxt_path, lnweights_path, useConvRelu):
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            # write the body of the backbone 1st
            features = model.backbone[0].features

            name = "data"
            parent_name = C.writeImageInput(name, [640, 480, 3, 1], lntxt)

            name = "stage1"
            modules = features.stage1
            parent_name = writeConvBN(modules[0].conv, name + "_convbn", parent_name, lntxt, lnweights,
                                      useConvRelu)
            parent_name = writeInvertedResidual(modules[1].conv, modules[1].use_res_connect,
                                                name + "_ir0", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage2"
            modules = features.stage2
            parent_name = writeInvertedResidual(modules[0].conv, modules[0].use_res_connect,
                                                name + "_ir0", parent_name, lntxt, lnweights, useConvRelu)
            stage2_out = writeInvertedResidual(modules[1].conv, modules[1].use_res_connect,
                                               name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage3"
            modules = features.stage3
            parent_name = writeInvertedResidual(modules[0].conv, modules[0].use_res_connect,
                                                name + "_ir0", stage2_out, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[1].conv, modules[1].use_res_connect,
                                                name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)
            stage3_out = writeInvertedResidual(modules[2].conv, modules[2].use_res_connect,
                                               name + "_ir2", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage4"
            modules = features.stage4
            parent_name = writeInvertedResidual(modules[0].conv, modules[0].use_res_connect,
                                                name + "_ir0", stage3_out, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[1].conv, modules[1].use_res_connect,
                                                name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[2].conv, modules[2].use_res_connect,
                                                name + "_ir2", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[3].conv, modules[3].use_res_connect,
                                                name + "_ir3", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[4].conv, modules[4].use_res_connect,
                                                name + "_ir4", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[5].conv, modules[5].use_res_connect,
                                                name + "_ir5", parent_name, lntxt, lnweights, useConvRelu)
            stage4_out = writeInvertedResidual(modules[6].conv, modules[6].use_res_connect,
                                               name + "_ir6", parent_name, lntxt, lnweights, useConvRelu)

            name = "stage5"
            modules = features.stage5
            parent_name = writeInvertedResidual(modules[0].conv, modules[0].use_res_connect,
                                                name + "_ir0", stage4_out, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[1].conv, modules[1].use_res_connect,
                                                name + "_ir1", parent_name, lntxt, lnweights, useConvRelu)
            parent_name = writeInvertedResidual(modules[2].conv, modules[2].use_res_connect,
                                                name + "_ir2", parent_name, lntxt, lnweights, useConvRelu)
            stage5_out = writeInvertedResidual(modules[3].conv, modules[3].use_res_connect,
                                               name + "_ir3", parent_name, lntxt, lnweights, useConvRelu)

            # next write the FPN part -- this could be done nicer in a for loop
            fpn = model.backbone[1]

            last_inner = C.writeConv(fpn.fpn_inner4, "fpn_inner4", stage5_out, lntxt, lnweights)
            result_4 = C.writeConv(fpn.fpn_layer4, "fpn_layer4", last_inner, lntxt, lnweights)

            inner_top_down = C.writeUpSampleNearest("stub", "fpn_up3", [2, 2], [0, 0, 0, 0], last_inner, lntxt)
            inner_lateral = C.writeConv(fpn.fpn_inner3, "fpn_inner3", stage4_out, lntxt, lnweights)
            last_inner = C.writeSum("fpn_sum3", inner_top_down, inner_lateral, lntxt)
            result_3 = C.writeConv(fpn.fpn_layer3, "fpn_layer3", last_inner, lntxt, lnweights)

            inner_top_down = C.writeUpSampleNearest("stub", "fpn_up2", [2, 2], [0, 0, 0, 0], last_inner, lntxt)
            inner_lateral = C.writeConv(fpn.fpn_inner2, "fpn_inner2", stage3_out, lntxt, lnweights)
            last_inner = C.writeSum("fpn_sum2", inner_top_down, inner_lateral, lntxt)
            result_2 = C.writeConv(fpn.fpn_layer2, "fpn_layer2", last_inner, lntxt, lnweights)

            inner_top_down = C.writeUpSampleNearest("stub", "fpn_up1", [2, 2], [0, 0, 0, 0], last_inner, lntxt)
            inner_lateral = C.writeConv(fpn.fpn_inner1, "fpn_inner1", stage2_out, lntxt, lnweights)
            last_inner = C.writeSum("fpn_sum1", inner_top_down, inner_lateral, lntxt)
            result_1 = C.writeConv(fpn.fpn_layer1, "fpn_layer1", last_inner, lntxt, lnweights)

            result_5 = C.writeMaxPoolFromParams("fpn_mp", 1, 2, 0, False, result_4, lntxt)

            # next write the RPN part -- this could be a separate file
            rpn = model.rpn.head
            name = "rpm"

            rpn1_conv = writeRPN_Conv(rpn.conv, name + "1", result_1, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, name + "1_cls_logits", rpn1_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, name + "1_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, name + "1_bbox_pred", rpn1_conv, lntxt, lnweights)

            rpn2_conv = writeRPN_Conv(rpn.conv, name + "2", result_2, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, name + "2_cls_logits", rpn2_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, name + "2_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, name + "2_bbox_pred", rpn2_conv, lntxt, lnweights)

            rpn3_conv = writeRPN_Conv(rpn.conv, name + "3", result_3, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, name + "3_cls_logits", rpn3_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, name + "3_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, name + "3_bbox_pred", rpn3_conv, lntxt, lnweights)

            rpn4_conv = writeRPN_Conv(rpn.conv, name + "4", result_4, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, name + "4_cls_logits", rpn4_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, name + "4_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, name + "4_bbox_pred", rpn4_conv, lntxt, lnweights)

            rpn5_conv = writeRPN_Conv(rpn.conv, name + "5", result_5, lntxt, lnweights, useConvRelu)
            C.writeConv(rpn.cls_logits, name + "5_cls_logits", rpn5_conv, lntxt, lnweights)
            C.writeSigmoid(rpn.cls_logits, name + "5_cls_logits_sm", "previous", lntxt, lnweights)
            C.writeConv(rpn.bbox_pred, name + "5_bbox_pred", rpn5_conv, lntxt, lnweights)

    print("write MobileRCNN base done")


def writeMobileRCNN_det(model, lntxt_path, lnweights_path, useConvRelu):
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            name = "det"

            #first write the box transform layer
            roialign = {"resolution" : 7, "channels" : 128, "sampling_ratio" : 2}
            C.writeRoiAlign(roialign, name + "_roialign", lntxt)

            #reshape so the fc works
            C.writeReshape([1, 1, roialign["resolution"] * roialign["resolution"] * roialign["channels"]],
                         name + "_reshape", "previous", lntxt)

            #next write what's left of the Box_Head
            parent_dims = [roialign["channels"], model.feature_extractor.fc6.out_features,
                           roialign["resolution"], roialign["resolution"]]

            C.writeFC(model.feature_extractor.fc6, name + "_fc6", "previous", lntxt, lnweights)
            C.writeReLU("", name + "_relu6", "previous", lntxt, lnweights)
            C.writeFC(model.feature_extractor.fc7, name + "_fc7", "previous", lntxt, lnweights)
            parent_name = C.writeReLU("", name + "_relu7", "previous", lntxt, lnweights)

            # the Box_Output part
            cls_score = C.writeFC(model.predictor.cls_score, name + "_cls_score", parent_name, lntxt, lnweights)
            bbox_pred = C.writeFC(model.predictor.bbox_pred, name + "_bbox_pred", parent_name, lntxt, lnweights)

            # the softmax over class scores (could potentially be remove)
            sm_cls_score = C.writeSoftMax(name + "_sm_cls_score", cls_score, lntxt)

            # the output nodes
            C.writeCopyOutput(name + "_copy_cls_score", sm_cls_score, lntxt)
            C.writeCopyOutput(name + "_copy_bbox_pred", bbox_pred, lntxt)

    print("write MobileRCNN det done")


def writeMobileRCNN_mask(model, lntxt_path, lnweights_path, useConvRelu):
    with open(lntxt_path, "w+") as lntxt:
        with open(lnweights_path, "wb+") as lnweights:
            name = "mask"

            # first write the box transform layer
            roialign = {"resolution": 14, "channels": 128, "sampling_ratio": 2}
            C.writeRoiAlign(roialign, name + "_roialign", lntxt)

            # write the mask_fcn
            C.writeConv(model.feature_extractor.mask_fcn1, name + "_fcn1", "previous", lntxt, lnweights)
            C.writeReLU("", name + "_relu_fcn1", "previous", lntxt, lnweights)

            C.writeConv(model.feature_extractor.mask_fcn2, name + "_fcn2", "previous", lntxt, lnweights)
            C.writeReLU("", name + "_relu_fcn2", "previous", lntxt, lnweights)

            C.writeConv(model.feature_extractor.mask_fcn3, name + "_fcn3", "previous", lntxt, lnweights)
            C.writeReLU("", name + "_relu_fcn3", "previous", lntxt, lnweights)

            C.writeConv(model.feature_extractor.mask_fcn4, name + "_fcn4", "previous", lntxt, lnweights)
            C.writeReLU("", name + "_relu_fcn4", "previous", lntxt, lnweights)

            # write the predictor
            C.writeConvTransposed(model.predictor.conv5_mask, name + "_conv5", "previous", lntxt, lnweights)
            C.writeReLU("", name + "_relu_conv5", "previous", lntxt, lnweights)

            C.writeConv(model.predictor.mask_fcn_logits, name + "_fcn_logits", "previous", lntxt, lnweights)

            # write post processor
            C.writeSigmoid("", name + "_sigmoid", "previous", lntxt, lnweights)

            # the output nodes
            C.writeCopyOutput(name + "_segmentation", "previous", lntxt)

    print("write MobileRCNN mask done")
