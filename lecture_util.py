from execute_util import link


def article_link(url: str) -> str:
    return link(title=" [article]", url=url)


def blog_link(url: str) -> str:
    return link(title=" [blog]", url=url)


def tweet_link(url: str) -> str:
    return link(title=" [tweet]", url=url)


def youtube_link(url: str) -> str:
    return link(title=" [video]", url=url)