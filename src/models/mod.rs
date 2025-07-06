pub mod names;

use std::path::PathBuf;

use reqwest::StatusCode;

use crate::models::names::models;

const REQUIRED_FILES: [&str; 5] = [
    "model.bin",
    "config.json",
    "preprocessor_config.json",
    "vocabulary.txt",
    "tokenizer.json",
];

pub struct ModelHandler {
    model_name: String,  // list of downloaded models
    models_dir: PathBuf, // path to the models directory
}

impl ModelHandler {
    pub async fn new(model_name: &str, models_dir: PathBuf) -> ModelHandler {
        let model_handler = ModelHandler {
            model_name: match models().get(&model_name.to_lowercase()) {
                Some(x) => x.to_string(),
                None => model_name.to_string(),
            },
            models_dir: models_dir,
        };

        if model_handler.is_model_existing() {
            return model_handler;
        }

        let _ = model_handler.setup_directory();
        let _ = model_handler.download_model().await;

        model_handler
    }

    /// setup the directory to which models will be downloaded.
    /// Sets a global vx
    ///
    /// # Returns
    ///
    /// * `Void` - directory is setup.
    fn setup_directory(&self) -> Result<(), std::io::Error> {
        if !self.is_model_existing() {
            let _ = std::fs::create_dir_all(self.get_model_dir())?;
        }
        Ok(())
    }

    pub fn get_model_dir(&self) -> PathBuf {
        let mut path = self.models_dir.clone();
        path.push(format!("{}/", self.model_name));
        path
    }

    pub fn get_model_bin(&self) -> PathBuf {
        let mut path = self.models_dir.clone();
        path.push(format!("{}/model.bin", self.model_name));
        path
    }

    pub fn is_model_existing(&self) -> bool {
        self.get_model_bin().exists()
    }

    /// Download the specified model.
    ///
    /// # Arguments
    ///
    /// * `model` - The name of the model to download.
    ///
    /// # Returns
    ///
    /// * `Void` - The model is downloaded to the models directory.
    async fn download_model(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.is_model_existing() {
            self.setup_directory()?;
        }

        let base_url = format!("https://huggingface.co/{}/resolve/main/", self.model_name);

        for required in REQUIRED_FILES {
            let mut response = reqwest::get(format!("{}{}", base_url, required)).await?;
            if response.content_length().unwrap() < 30 {
                response = reqwest::get(format!(
                    "https://huggingface.co/openai/whisper-base/resolve/main/{}",
                    required
                ))
                .await?;
            }

            let mut path = self.get_model_dir();
            path.push(required);
            let mut file = std::fs::File::create(path)?;
            let mut content = std::io::Cursor::new(response.bytes().await?);
            std::io::copy(&mut content, &mut file)?;
        }

        Ok(())
    }
}
