# Beams Commit Message Checker Docker Image

A docker image that runs a commit message quality checker based on Chris Beams' article ["How to Write a Git Commit Message"](https://cbea.ms/git-commit/).

## Build and publish

```sh
docker build -t beams-commit-message-checker .
docker tag beams-commit-message-checker ghcr.io/high-quality-commit-messages/beams-commit-message-checker:latest
docker tag beams-commit-message-checker ghcr.io/high-quality-commit-messages/beams-commit-message-checker:0.1.0
```
## Run

```sh
docker run -e COMMIT_MESSAGE="<commit-message>" -e FILENAMES="<filenames-separated-by-commas>" ghcr.io/commit-message-collective/beams-commit-message-checker:latest
```

Validation errors will be printed to `stdout`.
