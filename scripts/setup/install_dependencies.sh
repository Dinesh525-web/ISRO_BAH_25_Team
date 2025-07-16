#!/usr/bin/env bash
set -e

echo "🔧 Installing Python deps"
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "🔧 Installing Node deps"
cd src/frontend/react_app
npm ci
