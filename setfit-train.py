from datasets import load_dataset,concatenate_datasets
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

# Load the dataset
dataset = load_dataset("ag_news")

# create train dataset
seed=20
labels = 4
samples_per_label = 8
sampled_datasets = []
# find the number of samples per label
for i in range(labels):
    sampled_datasets.append(dataset["train"].filter(lambda x: x["label"] == i).shuffle(seed=seed).select(range(samples_per_label)))

# concatenate the sampled datasets
train_dataset = concatenate_datasets(sampled_datasets)

# create test dataset
test_dataset = dataset["test"]





# Load a SetFit model from Hub
model_id = "sentence-transformers/all-mpnet-base-v2"
model = SetFitModel.from_pretrained(model_id)

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=64,
    num_iterations=20, # The number of text pairs to generate for contrastive learning
    num_epochs=1, # The number of epochs to use for constrastive learning
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()

print(f"model used: {model_id}")
print(f"train dataset: {len(train_dataset)} samples")
print(f"accuracy: {metrics['accuracy']}")

#    model used: sentence-transformers/all-mpnet-base-v2
#    train dataset: 32 samples
#    accuracy: 0.8647368421052631


# model specfic hyperparameters
def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    model_id = params.get("model_id", "sentence-transformers/all-mpnet-base-v2")
    model_params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(model_id, **model_params)

# training hyperparameters
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        "num_iterations": trial.suggest_categorical("num_iterations", [5, 10, 20, 40, 80]),
        "seed": trial.suggest_int("seed", 1, 40),
        "max_iter": trial.suggest_int("max_iter", 50, 300),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        "model_id": trial.suggest_categorical(
            "model_id",
            [
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L12-v1",
            ],
        ),
    }




trainer = SetFitTrainer(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    model_init=model_init,
)

best_run = trainer.hyperparameter_search(direction="maximize", hp_space=hp_space, n_trials=100)

best_run.hyperparameters


trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
trainer.train()

metrics = trainer.evaluate()

print(f"model used: {best_run.hyperparameters['model_id']}")
print(f"train dataset: {len(train_dataset)} samples")
print(f"accuracy: {metrics['accuracy']}")
