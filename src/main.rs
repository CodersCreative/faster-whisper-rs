use ct2rs::Whisper;
use faster_whisper_rs::{audio::parse_audio_file, models::ModelHandler};
use std::error::Error;
use std::path::{Path, PathBuf};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let path = ModelHandler::new("tiny", PathBuf::from("models/")).await;
    let whisper = Whisper::new(path.get_model_dir(), Default::default())?;

    let samples = parse_audio_file(PathBuf::from("src/man.mp3"), whisper.sampling_rate() as u32);

    let res = whisper.generate(&samples, Some("en"), true, &Default::default())?;

    for r in res {
        println!("{}", r);
    }

    Ok(())
}
