#!/usr/bin/env python3
"""
Installation script for RAGAS and evaluation dependencies.
Run this script to install all required packages for RAGAS evaluation.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main installation function."""
    print("🚀 Installing RAGAS and evaluation dependencies...")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: You're not in a virtual environment.")
        print("   It's recommended to use a virtual environment.")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Change to the api directory where requirements.txt is located
    api_dir = os.path.join(os.path.dirname(__file__), 'api')
    if os.path.exists(api_dir):
        os.chdir(api_dir)
        print(f"📁 Changed to directory: {api_dir}")
    
    # Install requirements
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing requirements from requirements.txt"),
        ("pip install ragas==0.2.10", "Installing RAGAS specifically"),
        ("pip install pandas", "Installing pandas for data handling"),
    ]
    
    success_count = 0
    for command, description in commands:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"⚠️  Failed: {description}")
    
    print("\n" + "=" * 50)
    print(f"📊 Installation Summary: {success_count}/{len(commands)} commands succeeded")
    
    if success_count == len(commands):
        print("✅ All dependencies installed successfully!")
        print("\n🎯 Next steps:")
        print("1. Navigate to the api/evaluator directory")
        print("2. Run: python ragas_eval.py")
        print("3. The script will generate synthetic data for evaluation")
    else:
        print("❌ Some installations failed. Please check the errors above.")
        print("💡 Try running the failed commands manually.")
    
    # Verify RAGAS installation
    print("\n🔍 Verifying RAGAS installation...")
    try:
        import ragas
        print(f"✅ RAGAS version {ragas.__version__} is installed successfully!")
    except ImportError:
        print("❌ RAGAS is not properly installed. Please check the installation.")

if __name__ == "__main__":
    main()
