# F1ML: Formula 1 Machine Learning Analysis
---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Data Sources](#data-sources)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Future Plans](#future-plans)
7. [Acknowledgments](#acknowledgments)

---

## Introduction
This project is a passion project of mine that plays around with data in such a way that I find really interesting and fun, it aims to do the following:
- Extract, clean, and preprocess session data such as laps, telemetry, and weather.
- Build predictive models for race results using aggregated metrics.
- Perhaps provide a cool way to visualize the performance of different drivers on different tracks and conditions.

---

## Features
- **Comprehensive Data Cleaning:** Handles NaN values and irregularities in lap, telemetry, and weather data.
- **Custom Features:** Generates aggregated metrics like average sector times, pit indicators, and lap consistency measures.
- **Flexible Input Design:** Focuses on recent race data to predict future events without requiring extensive historical datasets.
- **Support for All F1 Sessions:** FP1, FP2, FP3, Qualifying, and Race sessions are analyzed and integrated seamlessly.

---

## Data Sources
The project uses multiple datasets, including:
- Lap data (sector times, speeds, positions).
- Telemetry data (car location, speed, and status).
- Weather conditions (track temperature, wind speed, and rain).
- Session results and driver details.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Aleksandar-Mladenoski/F1ML.git
   cd F1ML
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Prepare your datasets and place them in the `data` folder.
2. Run the preprocessing pipeline:
   ```bash
   python preprocess.py
   ```
3. Train a model:
   ```bash
   python train.py
   ```
4. Analyze predictions and results:
   ```bash
   python analyze.py
   ```

---

## Future Plans
- Build a sequential model ( such as a transformer ) that can better understand the spatial aspect of the F1 lap data
- Perhaps create some dashboard for predicting races in the future? We will see.

---

## Acknowledgments
This project is inspired by a passion for motorsports and machine learning. Special thanks to the F1 data community for making comprehensive datasets available.
