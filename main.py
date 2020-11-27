import os
from ops.argparser import argparser

if __name__ == "__main__":
    params = argparser()
    # print(params)
    if params['mode'] == -100:
        #extract the cell images from tif
        input_path = params['F']
        type = params['type']
        from Data_Processing.Generate_Segemented_Image import Generate_Segemented_Image,Generate_Segemented_Image_Update
        Generate_Segemented_Image_Update(input_path,params)
        
    elif params['mode']==0:
        #build model and training
        training_path=params['F']
        type=params['type']
        choose = params['choose']
        os.environ["CUDA_VISIBLE_DEVICES"] = choose
        from Training.Gen_Resnet_Model import Generate_Resnet_Model
        Generate_Resnet_Model(params,training_path,type)
    elif params['mode']==1:
        #evaluate the model
        testing_path = params['F']
        #testing_path=params['F1']
        type = params['type']
        model_path=params['M']
        choose = params['choose']
        os.environ["CUDA_VISIBLE_DEVICES"] = choose
        #evaluate the performance
        from Evaluate.Evaluate_Model import Evaluate_Model
        Evaluate_Model(params,testing_path,model_path,type)
    elif params['mode']==2:
        input_img_path=params['F']
        location_info_path=params['F1']
        model_path=params['M']
        if params['choose']!=-1:
            choose = params['choose']
            os.environ["CUDA_VISIBLE_DEVICES"] = choose
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] ="100"
        from Evaluate.Predict_Img import Predict_Img
        Predict_Img(params,input_img_path,location_info_path,model_path)

    elif params['mode']==3:
        #input an image to apply and save all images
        image_path=params['F']
        from Augment.Visualize_Image import Visualize_Image
        Visualize_Image(image_path)

    elif params['mode']==5:
        input_img_path = params['F']
        model_path = params['M']
        choose = params['choose']
        os.environ["CUDA_VISIBLE_DEVICES"] = choose
        from Evaluate.Segment_Predict_Img import Segment_Predict_Img

        Segment_Predict_Img(params, input_img_path,  model_path)
    elif params['mode']==6:
        input_img_path = params['F']
        model_path = params['M']
        if params['choose']!="-1":
            choose = params['choose']
            os.environ["CUDA_VISIBLE_DEVICES"] = choose
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] ="100"
            #if you want, you can also give the segmentation different filter size
        from Evaluate.CV2Segment_Predict_Img import CV2Segment_Predict_Img

        CV2Segment_Predict_Img(params, input_img_path, model_path)
    elif params['mode']==7:
        #evaluate our extraction performance
        testing_path = params['F']#the file that records all the evaluation files
        model_path = params['M']
        choose = params['choose']
        os.environ["CUDA_VISIBLE_DEVICES"] = choose
        # evaluate the performance
        from Evaluate.Evaluate_Segment_Model import Evaluate_Segment_Model
        Evaluate_Segment_Model(params, testing_path, model_path)
    elif params['mode']==8:
        input_img_dir = params['F']
        model_path = params['M']
        if params['choose'] != "-1":
            choose = params['choose']
            os.environ["CUDA_VISIBLE_DEVICES"] = choose
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "100"
            # if you want, you can also give the segmentation different filter size
        from Evaluate.CV2Segment_Predict_Img import CV2Segment_Predict_Img

        listfiles = os.listdir(input_img_dir)
        for item in listfiles:
            tmp_file = os.path.join(input_img_dir, item)
            CV2Segment_Predict_Img(params, tmp_file, model_path)



