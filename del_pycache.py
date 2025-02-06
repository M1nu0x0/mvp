import os
import shutil

def remove_pycache_and_pyc_files(root_dir="."):
    """하위 모든 경로에서 __pycache__ 폴더와 .pyc 파일을 삭제"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # __pycache__ 폴더 삭제
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path, ignore_errors=True)
            print(f"Deleted: {pycache_path}")
        
        # # .pyc 파일 삭제
        # for filename in filenames:
        #     if filename.endswith(".pyc"):
        #         file_path = os.path.join(dirpath, filename)
        #         os.remove(file_path)
        #         print(f"Deleted: {file_path}")

if __name__ == "__main__":
    remove_pycache_and_pyc_files()
    remove_pycache_and_pyc_files('/home/aisl/acaconda3/envs/mvp3/lib/python3.6/site-packages/numpy')
