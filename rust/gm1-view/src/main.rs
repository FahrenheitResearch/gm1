fn main() {
    let cli = match gm1_view::input::Cli::from_env() {
        Ok(cli) => cli,
        Err(error) => {
            eprintln!("gm1-view: {error}");
            std::process::exit(2);
        }
    };
    if let Some(root) = cli.verify_production_pack.as_deref() {
        match gm1_view::production_verify::verify_production_pack(root) {
            Ok(receipt) => {
                println!(
                    "{}",
                    serde_json::to_string(&receipt)
                        .expect("production-pack verification receipt is serializable")
                );
                return;
            }
            Err(error) => {
                let receipt =
                    gm1_view::production_verify::production_pack_verification_refusal(root, &error);
                println!(
                    "{}",
                    serde_json::to_string(&receipt)
                        .expect("production-pack refusal receipt is serializable")
                );
                std::process::exit(2);
            }
        }
    }
    let started = std::time::Instant::now();
    let loaded = if let Some(root) = cli.production_pack.as_deref() {
        gm1_view::production_playback::load_production_pack(root).and_then(|frame| {
            let residency = frame.renderer_residency()?;
            let dataset = frame.cloud_dataset()?;
            Ok((dataset, Some(residency), None))
        })
    } else if let Some(root) = cli.watch_production_series.as_deref() {
        gm1_view::production_playback::load_initial_production_series(root).and_then(|startup| {
            let residency = startup.playback.current().renderer_residency()?;
            let dataset = startup.playback.current().cloud_dataset()?;
            Ok((dataset, Some(residency), Some(startup)))
        })
    } else {
        match cli.watch_dir.as_deref() {
            Some(root) => gm1_view::watch::load_initial_watch_dataset(root)
                .map(|dataset| (dataset, None, None)),
            None => gm1_view::input::load_cli_dataset(&cli).map(|dataset| (dataset, None, None)),
        }
    };
    let (dataset, production, production_series) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            eprintln!("gm1-view: {error}");
            std::process::exit(2);
        }
    };
    eprintln!(
        "gm1-view: admitted {} step {} in {:.3}s ({})",
        dataset.artifact_kind,
        dataset.step,
        started.elapsed().as_secs_f64(),
        dataset.artifact_sha256,
    );
    match (production, production_series) {
        (Some(residency), Some(series)) => {
            if let Err(error) =
                gm1_view::app::run_production_series(cli, dataset, residency, series)
            {
                eprintln!("gm1-view: {error}");
                std::process::exit(2);
            }
        }
        (Some(residency), None) => {
            if let Err(error) = gm1_view::app::run_production(cli, dataset, residency) {
                eprintln!("gm1-view: {error}");
                std::process::exit(2);
            }
        }
        (None, None) => {
            if let Err(error) = gm1_view::app::run(cli, dataset) {
                eprintln!("gm1-view: {error}");
                std::process::exit(2);
            }
        }
        (None, Some(_)) => unreachable!("production series always supplies strict residency"),
    }
}
