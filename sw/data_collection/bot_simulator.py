import asyncio
import random
import time
from playwright.async_api import async_playwright

# Wikipedia's random-article endpoint
WIKI_RANDOM = "https://en.wikipedia.org/wiki/Special:Random"

# Scroll amount in pixels and delay between scroll bursts
FAST_SCROLL_RANGE = (300, 1200)
FAST_SCROLL_DELAY = (0.05, 0.25)

# A pool of random "gibberish" words used when simulating typing behavior
GIBBERISH_WORDS = [
    "test", "hello", "bot", "user", "login", "guest", "admin", "root",
    "abc", "abcd", "abc123", "123", "1234", "12345", "0000", "1111",
    "pass", "password", "passwd", "pwd", "login123", "user1", "guest1",

    "qwe", "qwer", "qwerty", "asdf", "zxcv", "aaa", "bbb", "ccc", "xyz",
    "foo", "bar", "foobar", "baz", "lol", "omg", "wtf", "spam", "ping",
    
    "admin123", "test123", "pw123", "letmein", "welcome", "default",
    "123qwe", "qwe123", "pass123", "myname", "useruser",
    "superuser", "root123", "qazwsx", "zaq12wsx", "1q2w3e",

    "select", "update", "info", "token", "email", "login", "session",
    "debug", "check", "verify", "captcha", "submit", "formdata",
    "search", "payload", "string", "flag", "value", "hidden",

    "apple", "banana", "cat", "dog", "mouse", "car", "train", "table",
    "sky", "blue", "rain", "sun", "moon", "night", "light",
    "water", "stone", "road", "music", "house", "page", "data",

    "xD3f", "nmk2", "lp9d", "tr55", "vvv1", "opq9", "xxzz", "bb12",
    "okok", "ok123", "nope", "why", "gggg", "testtest", "idk",
    "hmmm", "random", "zzz", "zzzzzz", "maybe", "tryagain",

    "thisisatest", "randomstring", "mypassword", "justchecking",
    "icanhaz", "someinput", "loremipsum", "hello123", "searching",
    "enteringtext", "botactivity", "generatedbybot",

    "john", "mike", "alex", "anna", "kate", "david", "tom", "emma",
    "jake", "lucy", "mark", "eva", "leo", "mia", "sam", "joel",

    "test@test.com", "user@example.com", "bot@bot.com",
    "random@mail.com", "hello@world.com",
]

def random_gibberish():
    """Return a random small text fragment."""
    if random.random() < 0.5:
        # 50% chance to pull from predefined pool
        return random.choice(GIBBERISH_WORDS)
    
    # Or random letters & digits
    length = random.randint(3, 10)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))

async def type_random_gibberish(page):
    """
    Randomly types a short 'bot-like' word.
    Sometimes targets a search bar, sometimes the body.
    """
    word = random_gibberish()

    async def safe_type(selector, text):
        """
        Type a random word into some element on the page or directly into the body.

        Behavior:
        - 40% chance → attempt typing in the Wikipedia search box
        - 20% chance → attempt typing in other <input> elements
        - Otherwise → type directly into the page using keyboard events

        Args:
            page (playwright Page): The active browser page.

        Returns:
            bool: Always True unless typing action itself fails.
        """
        try:
            el = await page.query_selector(selector)
            if not el:
                return False

            await el.click()

            await el.fill("")  
            await el.type(text, delay=random.randint(10, 60))
            
            # Randomly press Enter after typing with prob 0.3
            if random.random() < 0.3:
                el2 = await page.query_selector(selector)
                if el2:
                    await el2.press("Enter")
            return True

        except Exception:
            return False

    # Try typing smth into the search box with prob 0.4
    if random.random() < 0.4:
        ok = await safe_type("input[name='search']", word)
        if ok:
            return True
    
    # Try other input elements sometimes
    inputs = await page.query_selector_all("input[type='text'], input:not([type])")
    if inputs and random.random() < 0.2:
        selector = "input[type='text'], input:not([type])"
        ok = await safe_type(selector, word)
        if ok:
            return True

    try:
        await page.keyboard.type(word, delay=random.randint(5, 50))
        if random.random() < 0.2:
            await page.keyboard.press("Enter")
        if random.random() < 0.1:
            await page.keyboard.press("Tab")
    except:
        pass

    return True

async def fast_scroll(page):
    """
    Perform a rapid sequence of downward scrolls to simulate fast browsing.

    Args:
        page: Playwright Page object
    """
    for _ in range(random.randint(5, 20)):
        px = random.randint(*FAST_SCROLL_RANGE)
        await page.mouse.wheel(0, px)
        await asyncio.sleep(random.uniform(*FAST_SCROLL_DELAY))

async def click_random_link(page):
    """
    Click a random Wikipedia internal link on the page.

    Args:
        page: Playwright Page object.

    Returns:
        bool: True if a click was performed, otherwise False.
    """
    # Select internal wiki links only
    links = await page.query_selector_all("a[href^='/wiki/']")
    if not links:
        return False

    # Choose a random link to reduce predictability
    link = random.choice(links)
    box = await link.bounding_box()
    if not box:
        return False
    
    # Click the center of the element
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    await page.mouse.move(x, y)
    await page.mouse.click(x, y)
    
    return True

async def random_wait():
    await asyncio.sleep(random.uniform(0.2, 1.5))

async def wikipedia_bot():
    """
    Autonomous browsing bot for Wikipedia.

    Loop behavior:
      1. Load a random Wikipedia article
      2. Wait for DOM content to load
      3. Perform a fast scroll sequence
      4. Randomly choose to:
          - Scroll more
          - Click a random internal link
          - Type random text
      5. Repeat forever

    Notes:
        - Browser runs non-headless for visible activity.
        - Errors are intentionally ignored to keep loop running.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        while True:
            try:
                async with page.expect_navigation():
                    await page.goto(WIKI_RANDOM)
            except:
                pass
            
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except:
                pass

            await fast_scroll(page)

            action = random.choice(["scroll", "click", "type"])

            if action == "click":
                clicked = await click_random_link(page)
                if not clicked:
                    continue

            elif action == "type":
                await type_random_gibberish(page)

            await random_wait()

# Start bot
asyncio.run(wikipedia_bot())