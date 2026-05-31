# 🏥 JIO Healthlab

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![React](https://img.shields.io/badge/react-18.x-cyan)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange)
![Docker](https://img.shields.io/badge/docker-compose-blue)

**JIO Healthlab** is a modern, end-to-end laboratory diagnostics reporting system and management dashboard. Built with a high-performance **FastAPI** backend and a responsive **React/Vite** frontend, it helps lab administrators track urgent tests, patient details, branch loads, and diagnostic queues in real time.

---

## ✨ Key Features

- **Real-Time Dashboard**: Monitor totals, urgent work, status mix, test demand, and city-level loads.
- **Advanced Filtering & Sorting**: Track patient details, contact info, due dates, and specific tests.
- **Scalable Backend**: Built with FastAPI, backed by a robust MySQL database using SQLAlchemy and Alembic for migrations.
- **Production Ready**: Fully containerized using Docker and Docker Compose with Nginx reverse proxying.
- **1-Click AWS Deployment**: Includes automated bash scripts and Docker configs designed to run perfectly within the **AWS Free Tier** (t2.micro/t3.micro).

---

## 🏗️ Architecture Stack

- **Frontend**: React 18, Vite, TypeScript, TailwindCSS (optional), Lucide Icons
- **Backend**: FastAPI, Python 3.12, Uvicorn, Pydantic
- **Database**: MySQL 8.x, SQLAlchemy (ORM), Alembic (Migrations)
- **Deployment**: Docker, Docker Compose, Nginx

---

## 🚀 Quick Start (Local Docker)

The easiest way to run the project locally is via Docker.

1. **Clone the repo & copy environment variables**:
   ```bash
   git clone https://github.com/mohit6603/jio-healthlab.git
   cd jio-healthlab
   cp .env.example .env
   ```

2. **Start the application**:
   ```bash
   docker compose up --build
   ```

3. **Access the Application**:
   - 🌐 **Dashboard (Frontend)**: [http://localhost](http://localhost)
   - 📚 **API Swagger Docs**: [http://localhost/docs](http://localhost/docs)
   - 💚 **Backend Health Check**: [http://localhost/health](http://localhost/health)

*(The backend container will automatically run database migrations and seed the database with demo data.)*

---

## 🛠️ Local Development (Without Docker)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
alembic upgrade head
python -m app.seed
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*(The Vite dev server will automatically proxy API requests to the backend at `localhost:8000`.)*

---

## ☁️ Deployment (AWS Free Tier)

JIO Healthlab is designed to be easily deployed to AWS EC2 using Docker, completely within the Free Tier limits.

### Automated Deployment
We have provided a fully automated setup script for Ubuntu EC2 instances:
```bash
# On your EC2 Instance
git clone https://github.com/mohit6603/jio-healthlab.git
cd jio-healthlab
chmod +x deploy/aws-free-tier-setup.sh
./deploy/aws-free-tier-setup.sh
```

For detailed manual instructions (including setting up AWS RDS), check the deployment guide:
📖 **[AWS Deployment Guide](./deploy/aws-ec2-rds.md)**

---

## 📂 Project Structure

```text
jio-healthlab/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application & routes
│   │   ├── models.py        # SQLAlchemy database models
│   │   ├── schemas.py       # Pydantic validation schemas
│   │   ├── database.py      # DB engine and session configuration
│   │   └── seed.py          # Script to populate demo data
│   ├── alembic/             # Database migration scripts
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/                 # React components and views
│   │   ├── App.tsx          # Main Dashboard
│   │   └── api.ts           # API client configuration
│   ├── vite.config.ts       # Vite proxy setup
│   └── package.json         # Node.js dependencies
├── deploy/
│   ├── aws-ec2-rds.md       # AWS Documentation
│   └── aws-free-tier-setup.sh # Automated setup script
├── docker-compose.yml       # Local development stack
├── docker-compose.prod.yml  # Production deployment stack
└── .env.example             # Environment variables template
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
