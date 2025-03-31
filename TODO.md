[ ] Use SSL + iNaturalist to MAE the Whisper v3 large turbo's encoder
[ ] Use xeno-canto's labeled data to fine tune the encoder. (1 pooling layer, one optional dropout layer and a linear layer)
[ ] Segment the `train_audio` to filter out the human commentary. Potentially using Whisper to translate as well.
[ ] Integrate with Logfire, Pydantic and potentially Sentry for performance, program sequence logging and baseline.
[ ] During inference, use batch to process audio and run inference to maximise
the parallelism
