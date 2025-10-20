use ct2rs::Whisper;
use faster_whisper_rs::models::ModelHandlerBuilder;
use faster_whisper_rs::WhisperModel;
use faster_whisper_rs::{audio::parse_audio_file, models::ModelHandler};
use std::error::Error;
use std::path::{Path, PathBuf};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let whisper = WhisperModel::new_with_handler(
        ModelHandlerBuilder::default()
            .model_name("large-v3".to_string())
            .clone()
            .build()
            .await,
        Default::default(),
    )?;

    let samples = parse_audio_file(
        PathBuf::from("src/man.mp3"),
        whisper.model.sampling_rate() as u32,
    );

    let res = whisper.generate(&samples, Default::default())?;

    println!("{}", res.text);
    Ok(())
}
