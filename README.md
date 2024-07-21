# Operating System ⚙️
Windows or Ubuntu is compatible with this Service Learning Project.

# System Setup 🔧
> [!Warning]
> If it turns out you will be doing this in Ubuntu Virtual Machine from scratch, here are the following guideline to properly compensate for compatibility. **It is recommended to do it in Windows** instead to avoid memory leak, software crash, and kernel crashes (in ipynb).

### **In-Terminal**
- sudo apt-get update
- sudo apt-get install build-essential
- sudo apt-get install libgtk-3-dev
- sudo apt-get install libboost-all-dev
- sudo apt install libmpv-dev mpv

# Dependencies 💿
### Virtual  Environment

- pip install -r requirements.txt

### Dataset and Weights
- https://github.com/azra-dev/Facenet-Attendance-Checker-Using-Mindspore/releases/tag/v1.0.0

# Error Handling 🔧

### Git Error
During commit, sometimes an error of `RPC failed; HTTP 400 curl 22 The requested URL returned error: 400 Bad Request` will be shown likely due to committing something with a file size beyond 100MB. Run the command in the terminal to produce a .gitignore file to ignore large file size:
```bash
find . -size +100M | cat >> .gitignore
```
However, do remove the `./` prefix of each generated output to properly address the files to be ignored

### basicsr error
Upon running the sr-prototype.py program, an error will be recieved which comes from basicsr module due to deprication. The module is modifiable thankfully with the following step:

Open `your_venv/lib/python3.11/site-packages/basicsr/data/degradations.py` and on line 8, simply change:
```py
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```
to
```py
from torchvision.transforms.functional import rgb_to_grayscale
```