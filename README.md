# Deep Learning   - BMEVITMMA19-EN
BMEVITMMA19 task:Bot-guard

---

Team members:
- Csarkovszkij Artyjom Alekszandrovics - N852Z8
- Phaxay Thipphasone - FQ9TSP

---

## Project Destription
This project is a study of the  mouse usage patterns and training deep learning models for behavioral biometrics, particularily aims to make a significant contribution to the field of cybersecurity by developing and training advanced deep learning models capable of distinguishing between human and automated (bot) user behaviour based on mouse dynamics.

## 2. Datasets overview
### PMC
PMC is Dynamics Dataset (published via the National Library of Medicine, a U.S. government official source). This dataset contains 2,550 hours of active usage data from 24 users. (https://data.mendeley.com/datasets/w6cxr8yc7p/2) 

[Mendeley Data – PMC Mouse Dynamics](https://data.mendeley.com/datasets/w6cxr8yc7p/2) 


#### Structure
The dataset is organized into user folders, e.g.:
```
users/
├─ user1/
│ ├─ training/
│ │ ├─ session_*.csv
│ │ └─ ...
│ ├─ internal_tests/
│ └─ external_tests/
├─ user2/
└─ ...
```

- **Training**: multiple session recordings per user, provided as `.csv` files.  
- **Internal tests**: sessions reserved for within-dataset evaluation.  
- **External tests**: sessions reserved for cross-dataset/generalization testing.  

#### Data format
Each session file contains **timestamped mouse events**, including:  
- `client_timestamp` – time of event  
- `x`, `y` – screen coordinates  
- `button` – mouse button pressed (if any)  
- `state` – type of event (`Move`, `Click`, `Release`, `Scroll`)  
- `window` – active foreground application/window 

This format makes it possible to study continuous user interaction behaviour at a fine-grained level.

### OUR Dataset
In addition to the public PMC dataset, we created our own dataset using a custom Python logger (`collect_events.py`).  
This logger records **mouse and keyboard events** during normal computer usage and stores them in **JSONL** format.  

---

#### File organization
Our dataset is stored under:

```
data/raw/our/v1/
├─ key_events_*.jsonl
└─ mouse_events_*.jsonl
```

- **Key event files (`key_events_*.jsonl`)**  
  Contain only keyboard-related events (`key_down`, `key_up`) along with timing and foreground application info.  

- **Mouse event files (`mouse_events_*.jsonl`)**  
  Contain pointer movements, clicks, scrolls, releases, with screen coordinates and button metadata.  

This separation allows us to process **mouse-only data** (for compatibility with PMC) while still keeping **keyboard activity** available for future extensions of the model.

#### Event Schema
Each line in the JSONL file is a JSON object.  
There are **common fields** for all events, and some **type-specific fields**.

**Common fields (all events):**
- `schema_version` – version number of the logging schema.  
- `event_id` – unique identifier (UUID) for the event.  
- `session_id` – unique identifier (UUID) for the session.  
- `event_type` – type of event (`pointermove`, `click`, `scroll`, `key_down`, `key_up`).  
- `monotonic_ms` – monotonic timestamp in milliseconds (system performance counter, stable for measuring intervals).  
- `wall_time_ms` – wall-clock timestamp in milliseconds since Unix epoch (real-world time).  
- `foreground` – information about the active window/process at the time of the event:  
  - `pid` – process ID.  
  - `process_name` – name of the executable (e.g., `firefox.exe`, `Code.exe`).  

**Mouse-specific fields:**
- `x_screen`, `y_screen` – screen coordinates of the pointer.  
- `button` – mouse button (`Button.left`, `Button.right`, etc.).  
- `pressed` – `true` if the button is pressed, `false` if released.  
- `wheel_dx`, `wheel_dy` – scroll wheel delta (horizontal/vertical, only for scroll events).  

**Keyboard-specific fields:**
- `event_type` is `key_down` or `key_up`, indicating timing of key press/release.  

This schema it is abstracted to timing only (for privacy).  

---


#### Usage in this project
For **Milestone 1**, we only used **mouse movement events** (`pointermove`, `click`, `scroll`) from the collected sessions.  
This dataset complements PMC by providing **real-world, small-scale data** that we can directly control and extend.  

## 3. Results


## Instructions to run the solution

1. **Install dependencies**
    Recommended: Python ≥ 3.10. Install with:  

   ```
   pip install -r requirements.txt  
   ```

   Main dependencies:  
   - numpy  
   - pandas  
   - matplotlib  
   - scikit-learn  
   - tensorflow (for baseline model training)  
   - pynput, psutil, pywin32 (required for mouse_logger.py on Windows)  
2. **Prepare datasets**  
   All required datasets are downloaded and prepared automatically.  

   Run the notebook:
    
    ```
     00_Download_Datasets.ipynb
    ``` 

3. **(Optional) Collect your own data**  
   Run the provided logger script to generate new mouse/keyboard logs:  
   python scripts/mouse_logger.py  
   This will create new files under:  
    
    ```
   data/raw/our/v1/  
    ```

4. **Run the notebooks**  
   - 01_eda.ipynb → exploratory analysis and plots  
   - 02_prepare_and_split.ipynb → preprocessing, train/val/test splits, baseline training  
