# Security

LeakLens is a local review tool. It does not upload the file you scan.

## Reporting a problem

Open a GitHub issue for a crash, a missed category, or a false redaction. Do not attach the original photo, the review report, OCR dumps, or crops. Those files contain the secrets the tool found. Describe the category and, if you can, a synthetic example.

## Defaults

- `leaklens serve` binds to `127.0.0.1`. Use another `--host` only when you mean to expose the upload page.
- OCR dumps and review crops are off unless a config file turns them on.
- The YuNet face model may be downloaded once from the OpenCV model zoo. The media file is not part of that request.
- Report directories (`.leaklens`) hold detections and, when enabled, crops. Delete them when you are done.

## What this does not do

LeakLens does not guarantee that every secret is gone. A blur is only as good as the boxes it was given. Read `review.html` before sharing the redacted file.
