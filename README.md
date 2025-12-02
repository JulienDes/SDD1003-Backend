# SDD1003-Backend

## FastAPI Backend

### Virtual Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv env
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source ./env/bin/activate
     ```

3. Install dependencies inside the virtual environment:
   ```bash
  pip install requirements.txt
   ```
### Create a .env file
Create a .env file in the folder root, and add your mongoDB URI and OpenAI API key
  ```
  MONGODB_URI = 
  OPENAI_API_KEY = 
  ```
### Run the Backend

To start the FastAPI backend server, run the following command in your console (from the backend folder):

```bash
uvicorn main:app --reload
```

- The server will be available at: http://127.0.0.1:8000
- The interactive API docs are at: http://127.0.0.1:8000/docs
- The alternative docs are at: http://127.0.0.1:8000/redoc

Make sure your virtual environment is activated before running the command.