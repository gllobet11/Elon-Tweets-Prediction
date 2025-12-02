import pickle
import os

def inspect_pkl_file():
    """
    Loads a pickle file and inspects its content.
    """
    # Find the latest model file
    try:
        model_files = glob.glob('best_prophet_model_*.pkl')
        if not model_files:
            print("❌ No model .pkl file found.")
            return

        latest_model_path = max(model_files, key=os.path.getmtime)
        print(f"🔍 Inspecting file: {latest_model_path}")

        with open(latest_model_path, 'rb') as f:
            loaded_object = pickle.load(f)

        print(f"\n✅ Object loaded successfully.")
        print(f"   -> Type of loaded object: {type(loaded_object)}")

        if isinstance(loaded_object, dict):
            print("   -> Object is a dictionary. Keys:")
            for key in loaded_object.keys():
                print(f"      - {key}")
        else:
            print("\n   -> Content of the object:")
            print(loaded_object)

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    import glob
    inspect_pkl_file()
