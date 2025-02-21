from .Tool import *

def generate_mobilenet_v2_pipeline(
    resolution=320,
    num_classes=90,
    batch_size=128,
    learning_rate_base=0.08,
    total_steps=50000,
    fine_tune_checkpoint="PATH_TO_BE_CONFIGURED",
    label_map_path="PATH_TO_BE_CONFIGURED",
    train_input_path="PATH_TO_BE_CONFIGURED",
    vaild_input_path="PATH_TO_BE_CONFIGURED"
):
    """
    Tạo pipeline config cho SSD MobileNetV2 với các tham số chính.
    
    Các tham số chính:
      - resolution (int): Kích thước cố định cho image resizer (đề xuất: 320 hoặc 640).
      - num_classes (int): Số lớp đối tượng.
      - batch_size (int): Kích thước batch trong training.
      - learning_rate_base (float): Giá trị learning rate cơ sở cho hàm cosine decay.
      - total_steps (int): Tổng số bước training.
      - fine_tune_checkpoint (str): Đường dẫn checkpoint để fine tune.
      - label_map_path (str): Đường dẫn file label map.
      - input_path (str): Đường dẫn file TFRecord.
    
    Lưu ý: Các thông số nâng cao (như hyperparams, cấu trúc box_coder, matcher, …)
    đã được cố định để tránh ảnh hưởng đến quá trình fine tune.
    
    Trả về:
      Một chuỗi (string) chứa cấu hình pipeline.
    """
    # Lưu ý: Các dấu ngoặc nhọn xuất hiện trong chuỗi được “escape” bằng cách dùng {{ và }}
    config = f"""model {{
  ssd {{
    num_classes: {num_classes}
    image_resizer {{
      fixed_shape_resizer {{
        height: {resolution}
        width: {resolution}
      }}
    }}
    feature_extractor {{
      type: "ssd_mobilenet_v2_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {{
        regularizer {{
          l2_regularizer {{
            weight: 3.9999998989515007e-05
          }}
        }}
        initializer {{
          random_normal_initializer {{
            mean: 0.0
            stddev: 0.009999999776482582
          }}
        }}
        activation: RELU_6
        batch_norm {{
          decay: 0.996999979019165
          scale: true
          epsilon: 0.0010000000474974513
        }}
      }}
      use_depthwise: true
      override_base_feature_extractor_hyperparams: true
      fpn {{
        min_level: 3
        max_level: 7
        additional_layer_depth: 128
      }}
    }}
    box_coder {{
      faster_rcnn_box_coder {{
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }}
    }}
    matcher {{
      argmax_matcher {{
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }}
    }}
    similarity_calculator {{
      iou_similarity {{
      }}
    }}
    box_predictor {{
      weight_shared_convolutional_box_predictor {{
        conv_hyperparams {{
          regularizer {{
            l2_regularizer {{
              weight: 3.9999998989515007e-05
            }}
          }}
          initializer {{
            random_normal_initializer {{
              mean: 0.0
              stddev: 0.009999999776482582
            }}
          }}
          activation: RELU_6
          batch_norm {{
            decay: 0.996999979019165
            scale: true
            epsilon: 0.0010000000474974513
          }}
        }}
        depth: 128
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
        share_prediction_tower: true
        use_depthwise: true
      }}
    }}
    anchor_generator {{
      multiscale_anchor_generator {{
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }}
    }}
    post_processing {{
      batch_non_max_suppression {{
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
      }}
      score_converter: SIGMOID
    }}
    normalize_loss_by_num_matches: true
    loss {{
      localization_loss {{
        weighted_smooth_l1 {{
        }}
      }}
      classification_loss {{
        weighted_sigmoid_focal {{
          gamma: 2.0
          alpha: 0.25
        }}
      }}
      classification_weight: 1.0
      localization_weight: 1.0
    }}
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }}
}} 
train_config {{
  batch_size: {batch_size}
  data_augmentation_options {{
    random_horizontal_flip {{
    }}
  }}
  data_augmentation_options {{
    random_crop_image {{
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }}
  }}
  optimizer {{
    momentum_optimizer {{
      learning_rate {{
        cosine_decay_learning_rate {{
          learning_rate_base: {learning_rate_base}
          total_steps: {total_steps}
          warmup_learning_rate: 0.026666000485420227
          warmup_steps: 1000
        }}
      }}
      momentum_optimizer_value: 0.8999999761581421
    }}
    use_moving_average: false
  }}
  fine_tune_checkpoint: "{normalize_path(fine_tune_checkpoint)}"
  num_steps: {total_steps}
  startup_delay_steps: 0.0
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
}} 
train_input_reader {{
  label_map_path: "{normalize_path(label_map_path)}"
  tf_record_input_reader {{
    input_path: "{normalize_path(train_input_path)}"
  }}
}} 
eval_config {{
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}} 
eval_input_reader {{
  label_map_path: "{normalize_path(label_map_path)}"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {{
    input_path: "{normalize_path(vaild_input_path)}"
  }}
}}"""
    return config


def normalize_path(path: str) -> str:
    """
    Chuyển đổi dấu '\' thành '/' trong đường dẫn để đảm bảo tính nhất quán.
    
    Tham số:
    - path (str): Đường dẫn cần chuẩn hóa.
    
    Trả về:
    - str: Đường dẫn đã chuẩn hóa.
    """
    return path.replace("\\", "/")



