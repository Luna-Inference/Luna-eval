from lighteval.pipeline import Pipeline
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import PipelineParameters, ParallelismManager
from lighteval.logging.evaluation_tracker import EvaluationTracker
# from lighteval.utils.utils import EnvConfig

# Set up where to save results and what to log
tracker = EvaluationTracker(output_dir="./results", save_details=True)

# Configure pipeline: using Accelerate (local) launcher and cache directory
pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.ACCELERATE,
    # env_config=EnvConfig(cache_dir="tmp/"),
    # override_batch_size=1,   # for demo, process 1 at a time
    max_samples=10           # for demo, limit to 10 samples per task
)

# Specify the model configuration (here, a transformers model)
model_config = TransformersModelConfig(
    model_name="openai-community/gpt2",
    dtype="float16"
)

# Define the task (TruthfulQA, zero-shot)
task = "leaderboard|truthfulqa:mc|0|0"

# Create and run the evaluation pipeline
pipeline = Pipeline(
    tasks=task,
    model_config=model_config,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=tracker
)
pipeline.evaluate()
pipeline.save_and_push_results()   # Save results locally (and push to Hub if configured)
pipeline.show_results()           # Display summary metrics
