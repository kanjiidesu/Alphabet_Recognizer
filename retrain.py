#!env/bin/python3

from letter_recognition import train_model, MODEL_FILENAME


def main() -> None:
    train_model(MODEL_FILENAME)


if __name__ == '__main__':
    main()
