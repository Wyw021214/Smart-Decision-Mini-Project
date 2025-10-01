# TSP & VRP Optimization with Metaheuristics

This repository provides Python implementations of several metaheuristic optimization algorithms applied to two classical combinatorial optimization problems:

- **Traveling Salesman Problem (TSP)**  
- **Vehicle Routing Problem (VRP)**

The implemented algorithms include:

- **SA** – Simulated Annealing  
- **ACO** – Ant Colony Optimization  
- **Tabu** – Tabu Search  
- **GA** – Genetic Algorithm  

These algorithms are widely used to tackle NP-hard optimization problems where exact methods are computationally infeasible for large instances.

---

## 🔗 Interactive Demo

We provide an interactive web application where you can test and visualize the algorithms directly:  

👉 [Streamlit Web App](https://wyw021214-smart-decision-mini-project-app-streamlit-q6qsmm.streamlit.app/)

---

## 📂 Features

- Multiple metaheuristic optimization algorithms for TSP and VRP
- Modular and extensible Python implementation
- Support for custom instance generation and real-world datasets
- Interactive visualizations of routes and convergence
- Web interface for experimentation and comparison

---

## ⚙️ Algorithms

### 1. Simulated Annealing (SA)
A probabilistic optimization technique inspired by the physical process of annealing in metallurgy. It explores the solution space by accepting not only improvements but also worse solutions with a probability that decreases over time.

### 2. Ant Colony Optimization (ACO)
A swarm intelligence algorithm inspired by the foraging behavior of ants. Artificial ants deposit and follow virtual pheromone trails to collaboratively construct good solutions.

### 3. Tabu Search (Tabu)
A local search method that uses memory structures (the **tabu list**) to avoid cycling back to previously visited solutions and to encourage exploration.

### 4. Genetic Algorithm (GA)
An evolutionary algorithm that simulates natural selection by applying crossover, mutation, and selection operators to evolve a population of solutions over generations.

---

Install dependencies:

pip install -r requirements.txt


Run experiments:

python run_experiments.py


Launch the Streamlit app locally:

streamlit run app.py

