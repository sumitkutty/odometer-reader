import os
import torch

def base_opts(parser):
    parser.add_argument('--images_path', required = True, help = "Path to input test images")
    parser.add_argument('--image_folder', default = "temp", help='path to image_folder which contains text image')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--saved_model', default="trained_models/best_ocr_model.pth", help="path to saved_model to evaluation")
    parser.add_argument('--generate_csv',action='store_true', help="To generate csv results file")
                        
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789.', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    
    
def set_cfg(cfg):
    # Save model for testing\
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.OUTPUT_DIR = 'trained_models'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    return cfg

def make_integer(pts):
            pts = [int(pt) for pt in pts]
            return pts