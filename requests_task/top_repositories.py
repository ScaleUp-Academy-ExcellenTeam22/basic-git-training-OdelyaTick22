import requests


def top_repositories(language: str, number_of_repos: int) -> None:
    """
    A function that requests the top repositories from GitHub, by programming language,
    ordered by the number of stars each repository has.
    :param language: The programming language to filter by.
    :param number_of_repos: The number of repositories to return.
    :return: None
    """
    url = f'https://api.github.com/search/repositories?q=language:{language}&sort=stars'
    response = requests.get(url)
    response_dict = response.json()
    repos_dicts = response_dict['items']
    for i in range(number_of_repos):
        repo_dict = repos_dicts[i]
        lang = language.capitalize()
        print(f"-> {repo_dict['name']} is a {lang} repo with {repo_dict['stargazers_count']} stars.")


if __name__ == "__main__":
    top_repositories("python", 20)
