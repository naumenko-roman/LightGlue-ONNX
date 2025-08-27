import onnx
import onnx.shape_inference
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def infer_shapes_with_validation(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input model file not found: {input_path}")
        
        logger.info(f"Loading model from {input_path}")
        model = onnx.load(input_path)
        
        onnx.checker.check_model(model)
        logger.info("Model validation passed")
        
        logger.info("Performing shape inference...")
        model_with_shapes = onnx.shape_inference.infer_shapes(model)
        
        onnx.checker.check_model(model_with_shapes)
        
        logger.info(f"Saving model with inferred shapes to {output_path}")
        onnx.save(model_with_shapes, output_path)
        
        logger.info("Shape inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shape inference: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    
    args = parser.parse_args()

    output_path = args.input.replace(".onnx", "_shapes.onnx")

    infer_shapes_with_validation(
        args.input,
        output_path
    )