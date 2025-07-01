import requests
import re
import os
import urllib.parse

def get_pdf_arxiv(web_site, path):
    try:
        response = requests.get(web_site, stream=True)
        if response.status_code == 200:

            if 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded PDF to {path}")
            else:
                print(f"Not a PDF file at {web_site}")
        else:
            print(f"Failed to download from {web_site}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading from {web_site}: {e}")

def search_arxiv(paper_title):
    query = paper_title.replace(" ", "+")
    query = urllib.parse.quote(query, safe=":/+[]") 
    print("query:", query)
    search_url = f"https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=-announced_date_first&size=50"
    print("search_url:", search_url)
    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            page = response.text
            arxiv_link = re.findall(r'<a href="(https://arxiv.org/pdf/\S+)"[^>]*>pdf</a>', page, re.S)
            print("arxiv_link:", arxiv_link)
            if arxiv_link:
                arxiv_url = arxiv_link[0]
                print("arxiv_url:", arxiv_url)
                return arxiv_url
            else:
                print("No PDF link found for the paper.")
                return None
        else:
            print(f"Failed to search arxiv. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error searching arxiv for {paper_title}: {e}")
        return None

path_dir = "papers/"
if not os.path.exists(path_dir):
    os.makedirs(path_dir)

for j in range(0, 38):  # 22-25
    print("j=",j)
    req = urllib.request.Request(f'https://dblp.uni-trier.de/search//publ/inc?q=neural%20architecture%20search&s=ydvspc&h=30&b={j}')
    response = urllib.request.urlopen(req)
    the_page = response.read().decode('utf-8')


    paper_title = re.findall('<span class="title" itemprop="name">(.*?)</span>', the_page, re.S)
    

    paper_web = re.findall('view</b></p><ul><li class="ee"><a href="(.*?)" itemprop="url">', the_page, re.S)

    for i in range(len(paper_title)):
        paper_title[i] = paper_title[i].rstrip('.')
        print(f"Searching for: {paper_title[i]}")

        arxiv_url = search_arxiv(paper_title[i])
        
        if arxiv_url:
            print(f"Found paper on arxiv: {arxiv_url}")

            sanitized_title = paper_title[i].replace('"', '').replace(" ", "_").replace(":", "_").replace("?", "_").replace("<sub>", "_").replace("</sub>", "_").replace("<sup>", "_").replace("</sup>", "_")
            pdf_path = os.path.join(path_dir, sanitized_title + ".pdf")

            if not os.path.exists(pdf_path):
                get_pdf_arxiv(arxiv_url, pdf_path)
            else:
                print(f"PDF already exists for {paper_title[i]}")
        else:
            print(f"Could not find paper on arxiv for {paper_title[i]}")

