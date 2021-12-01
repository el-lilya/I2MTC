import pandas as pd


def main():
    df = pd.read_json('data/clip/clipsubset.json')
    urls = df['url']
    urls.to_csv('data/clip/urls.txt', index=False, header=False)
    print(urls)


if __name__ == "__main__":
    main()
