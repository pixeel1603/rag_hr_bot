import requests
import os

from bs4 import BeautifulSoup


def confluence_login(login = '',password='', login_base_url='' ):
    login_url = f'{login_base_url}/dologin.action'

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'DNT': '1',
        'Origin': login_base_url,
        'Pragma': 'no-cache',
        'Referer': f'{login_base_url}/login.action?logout=true',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-gpc': '1'
    }

    login_payload = {
        'os_username': login,
        'os_password': password,
        'login': 'Log in',
        'os_destination': ''
    }

    session = requests.Session()

    response = session.post(login_url, headers=headers, data=login_payload)

    if response.status_code == 200:
        print("Login successful")
    else:
        print("Login failed", response.status_code)
        return None

    return session


def fetch_confluence_page(session, pageId, login_base_url=''):
    if pageId.isdigit():
        page_url = f'{login_base_url}/pages/viewpage.action?pageId={pageId}'
    else:
        page_url = f'{login_base_url}/display/HR/{pageId}'

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'DNT': '1',
        'Pragma': 'no-cache',
        'Referer': login_base_url,
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-gpc': '1'
    }

    response = session.get(page_url, headers=headers)

    if response.status_code == 200:
        print("Page fetched successfully")
        return response.text
    else:
        print("Failed to fetch page", response.status_code)
        return None


def get_page(pageId, output_dir='docs'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, pageId + ".html")

    # Log in to Confluence
    session = confluence_login()
    if session:
        # Fetch the Confluence page
        page_html = fetch_confluence_page(session, pageId)
        if page_html:
            # Parse the HTML content
            soup = BeautifulSoup(page_html, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.string if title_tag else 'No Title'

            # Extract main content
            main_content = soup.find('div', id='main-content')

            # Create new soup with the extracted content
            new_soup = BeautifulSoup('<html><head></head><body></body></html>', 'html.parser')
            new_soup.head.append(new_soup.new_tag('title'))
            new_soup.title.string = title
            if main_content:
                new_soup.body.append(main_content)

            # Write the new HTML content to a file (overwrite if it exists)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_soup.prettify())

            return output_dir + "/" + pageId + ".html"
        else:
            raise Exception("Failed to fetch the page HTML.")
    else:
        raise Exception("Failed to login to Confluence.")
