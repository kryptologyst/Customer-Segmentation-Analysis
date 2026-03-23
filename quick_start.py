#!/usr/bin/env python3
"""
Quick start script for Customer Segmentation Analysis.

This script provides a simple way to run the complete analysis pipeline
and launch the interactive demo.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main function to provide quick start options."""
    
    print("=" * 60)
    print("CUSTOMER SEGMENTATION ANALYSIS - QUICK START")
    print("=" * 60)
    print()
    print("Choose an option:")
    print("1. Run complete analysis pipeline")
    print("2. Launch interactive Streamlit demo")
    print("3. Run simple example (original code)")
    print("4. Run tests")
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\nRunning complete analysis pipeline...")
                print("This will generate data, train models, and create visualizations.")
                print("Results will be saved to the assets/ directory.")
                print()
                
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        subprocess.run([sys.executable, "scripts/run_analysis.py"], check=True)
                        print("\nAnalysis completed successfully!")
                        print("Check the assets/ directory for results and visualizations.")
                    except subprocess.CalledProcessError as e:
                        print(f"\nError running analysis: {e}")
                    except FileNotFoundError:
                        print("\nError: scripts/run_analysis.py not found")
                break
                
            elif choice == "2":
                print("\nLaunching interactive Streamlit demo...")
                print("The demo will open in your web browser.")
                print("Press Ctrl+C to stop the demo.")
                print()
                
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    try:
                        subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"\nError launching demo: {e}")
                    except FileNotFoundError:
                        print("\nError: streamlit not found. Install with: pip install streamlit")
                break
                
            elif choice == "3":
                print("\nRunning simple example...")
                print("This demonstrates the original basic approach.")
                print()
                
                try:
                    subprocess.run([sys.executable, "0801.py"], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"\nError running example: {e}")
                except FileNotFoundError:
                    print("\nError: 0801.py not found")
                break
                
            elif choice == "4":
                print("\nRunning tests...")
                print("This will run the test suite to verify functionality.")
                print()
                
                try:
                    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
                    print("\nAll tests passed!")
                except subprocess.CalledProcessError as e:
                    print(f"\nSome tests failed: {e}")
                except FileNotFoundError:
                    print("\nError: pytest not found. Install with: pip install pytest")
                break
                
            elif choice == "5":
                print("\nGoodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break
    
    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("For detailed documentation, see DISCLAIMER.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
