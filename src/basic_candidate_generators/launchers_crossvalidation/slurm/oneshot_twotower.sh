#!/#!/bin/bash
#SBATCH --job-name=cv_twotower_session
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4       # Cambia se il data loader ha bisogno di più core
#SBATCH --mem=16GB              # Incrementato: i modelli Two-Tower richiedono più RAM rispetto a iALS per i vettori di embedding
#SBATCH --time=12:00:00
#SBATCH --output=logs/cv_twotower_session-%j.out
#SBATCH --error=logs/cv_twotower_session-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --requeue              # Mantiene il comportamento di restart automatico in caso di blackout

# Richiesta della GPU (Fondamentale per il Two-Tower)
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM_16GB  # Cambia in GPU_MEM_32GB se il batch size o gli embedding sono massicci

REPO=/SLURM_shared/st_challenge26/nicolo/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

# Sincronizza l'ambiente uv (eseguito sul nodo di calcolo assegnato)
uv lock && uv sync

# Esecuzione con srun abilitando il binding della CPU raccomandato dalla guida
srun --unbuffered --cpu-bind=verbose uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model twotower \
    --urm_mode session

echo "[$(date)] done"