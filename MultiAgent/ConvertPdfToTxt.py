from grobid_client_python.grobid_client.grobid_client import GrobidClient
import glob
import os
if __name__ == "__main__":
    client = GrobidClient(config_path="./grobid_client_python/config.json")
    
    # 循环，直到 xml_count 等于 457
    while True:
        client.process("processFulltextDocument", "papers/", output="txts/", consolidate_citations=True, tei_coordinates=True, force=True)
        
        # 删除 txt 文件
        txt_files = glob.glob(os.path.join("txts", '*.txt'))
        for file in txt_files:
            os.remove(file)
        
        # 统计 xml 文件数量
        xml_count = 0
        for file in os.listdir("txts"):
            if file.endswith('.xml'):
                xml_count += 1
        
        print(f"There are {xml_count} XML files in the folder.")
        
        # 如果 xml_count 达到 457，就退出循环
        if xml_count == 460:
            print("Reached the target xml_count of 457. Exiting.")
            break
