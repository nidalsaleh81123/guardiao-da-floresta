"""
🌿 Guardião da Floresta - Edge Deployment Module
==================================================
Scripts and utilities for deploying on Raspberry Pi, Jetson Nano, and other edge devices.

Competition: Build with Gemma: Amazon Eco-Hack
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class EdgeDeployer:
    """Deploy Guardião da Floresta on edge devices"""

    def __init__(self, project_dir: str = ".", model_dir: str = "./models"):
        self.project_dir = Path(project_dir)
        self.model_dir = Path(model_dir)
        self.config = self._load_deployment_config()

    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        return {
            "supported_devices": {
                "raspberry_pi_4": {
                    "ram_gb": 4,
                    "storage_gb": 32,
                    "cpu_cores": 4,
                    "gpu": "VideoCore VI",
                    "recommended_model": "gemma-4-4B-it-Q4_0-GGUF",
                    "expected_tokens_per_sec": 8
                },
                "raspberry_pi_5": {
                    "ram_gb": 8,
                    "storage_gb": 64,
                    "cpu_cores": 4,
                    "gpu": "VideoCore VII",
                    "recommended_model": "gemma-4-4B-it-Q4_0-GGUF",
                    "expected_tokens_per_sec": 15
                },
                "jetson_nano": {
                    "ram_gb": 4,
                    "storage_gb": 64,
                    "cpu_cores": 4,
                    "gpu": "128-core Maxwell",
                    "recommended_model": "gemma-4-4B-it-Q4_0-GGUF",
                    "expected_tokens_per_sec": 12
                },
                "coral_dev_board": {
                    "ram_gb": 1,
                    "storage_gb": 8,
                    "cpu_cores": 4,
                    "gpu": "Edge TPU",
                    "recommended_model": "gemma-4-4B-it-Q4_0-GGUF",
                    "expected_tokens_per_sec": 5
                }
            },
            "optimization_settings": {
                "quantization": "Q4_0",
                "thread_count": 4,
                "batch_size": 1,
                "context_length": 2048,
                "temperature": 0.7
            }
        }

    def prepare_model_for_edge(self, source_model: str = "google/gemma-4-4B-it",
                               output_format: str = "gguf") -> str:
        """
        Convert model to edge-optimized format

        Args:
            source_model: HuggingFace model name
            output_format: Target format (gguf, onnx, tflite)

        Returns:
            Path to optimized model
        """
        print(f"🔄 Converting {source_model} to {output_format.upper()}...")

        output_dir = self.model_dir / f"gemma-4-edge-{output_format}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "gguf":
            return self._convert_to_gguf(source_model, output_dir)
        elif output_format == "onnx":
            return self._convert_to_onnx(source_model, output_dir)
        elif output_format == "tflite":
            return self._convert_to_tflite(source_model, output_dir)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _convert_to_gguf(self, source_model: str, output_dir: Path) -> str:
        """Convert to GGUF format for llama.cpp"""
        try:
            # Install llama.cpp conversion tools
            subprocess.run([
                "pip", "install", "llama-cpp-python", "-q"
            ], check=True)

            # Download and convert
            print("📥 Downloading model...")
            subprocess.run([
                "huggingface-cli", "download",
                source_model,
                "--local-dir", str(output_dir / "hf_model"),
                "--local-dir-use-symlinks", "False"
            ], check=True)

            print("⚙️ Converting to GGUF...")
            # Use llama.cpp convert script
            convert_script = Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "llama_cpp"

            # Alternative: Use direct download of pre-converted GGUF
            print("📥 Downloading pre-converted GGUF model...")
            subprocess.run([
                "huggingface-cli", "download",
                "google/gemma-4-4B-it-QAT-Q4_0-GGUF",
                "--local-dir", str(output_dir),
                "--local-dir-use-symlinks", "False"
            ], check=True)

            print(f"✅ GGUF model ready at: {output_dir}")
            return str(output_dir)

        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed: {e}")
            # Fallback: create placeholder
            return self._create_model_placeholder(output_dir)

    def _convert_to_onnx(self, source_model: str, output_dir: Path) -> str:
        """Convert to ONNX format"""
        print("⚙️ Converting to ONNX...")

        try:
            subprocess.run([
                "python", "-m", "transformers.onnx",
                "--model", source_model,
                "--feature", "causal-lm",
                str(output_dir)
            ], check=True)

            print(f"✅ ONNX model ready at: {output_dir}")
            return str(output_dir)

        except subprocess.CalledProcessError:
            print("⚠️ ONNX conversion failed, using fallback")
            return self._create_model_placeholder(output_dir)

    def _convert_to_tflite(self, source_model: str, output_dir: Path) -> str:
        """Convert to TensorFlow Lite"""
        print("⚙️ Converting to TFLite...")
        print("⚠️ TFLite conversion requires TensorFlow. Using placeholder.")
        return self._create_model_placeholder(output_dir)

    def _create_model_placeholder(self, output_dir: Path) -> str:
        """Create a placeholder for model directory"""
        placeholder = output_dir / "README.txt"
        placeholder.write_text(
            "Model files should be placed here.\n"
            "Download from: https://huggingface.co/google/gemma-4-4B-it-QAT-Q4_0-GGUF\n"
        )
        return str(output_dir)

    def create_systemd_service(self, device_type: str = "raspberry_pi") -> str:
        """Create systemd service for auto-start"""
        service_content = f"""[Unit]
Description=Guardião da Floresta - Amazon Conservation AI
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/guardiao-da-floresta
Environment=PYTHONPATH=/home/pi/guardiao-da-floresta
Environment=MODEL_PATH=/home/pi/guardiao-da-floresta/models
Environment=OMP_NUM_THREADS=4
ExecStart=/home/pi/guardiao-da-floresta/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

        service_path = self.project_dir / "guardiao.service"
        service_path.write_text(service_content)

        print(f"✅ Systemd service created: {service_path}")
        print("📋 To install:")
        print("   sudo cp guardiao.service /etc/systemd/system/")
        print("   sudo systemctl enable guardiao")
        print("   sudo systemctl start guardiao")

        return str(service_path)

    def create_install_script(self, device_type: str = "raspberry_pi_4") -> str:
        """Create installation script for target device"""
        script = f"""#!/bin/bash
# Guardião da Floresta - Installation Script
# Device: {device_type}
# Target: Raspberry Pi OS / Debian-based Linux

set -e

echo "🌿 Guardião da Floresta - Installation"
echo "========================================"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    htop

# Create project directory
INSTALL_DIR="/home/pi/guardiao-da-floresta"
echo "📁 Creating project directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

# Clone repository (or copy files)
echo "📥 Downloading project files..."
# git clone https://github.com/YOUR_USERNAME/guardiao-da-floresta.git .
# For now, copy from current directory
cp -r {self.project_dir}/* $INSTALL_DIR/

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Download model (if not present)
if [ ! -d "models/gemma-4-4B" ]; then
    echo "🤖 Downloading Gemma 4 model..."
    huggingface-cli download google/gemma-4-4B-it-QAT-Q4_0-GGUF \
        --local-dir ./models/gemma-4-4B
fi

# Create systemd service
echo "⚙️ Creating system service..."
sudo cp guardiao.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable guardiao

# Create desktop shortcut
echo "🖥️ Creating desktop shortcut..."
cat > /home/pi/Desktop/Guardiao.desktop << 'EOF'
[Desktop Entry]
Name=Guardião da Floresta
Comment=Amazon Conservation AI
Exec=chromium-browser http://localhost:7860
Type=Application
Icon=utilities-terminal
EOF
chmod +x /home/pi/Desktop/Guardiao.desktop

echo ""
echo "✅ Installation complete!"
echo "🚀 Starting Guardião da Floresta..."
sudo systemctl start guardiao

echo ""
echo "📊 Access the application at: http://localhost:7860"
echo "📊 Or from other devices: http://$(hostname -I | awk '{{print $1}}'):7860"
echo ""
echo "📝 Useful commands:"
echo "   sudo systemctl status guardiao    # Check status"
echo "   sudo systemctl restart guardiao   # Restart"
echo "   sudo journalctl -u guardiao -f    # View logs"
"""

        script_path = self.project_dir / "install.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        print(f"✅ Installation script created: {script_path}")
        return str(script_path)

    def create_solar_power_config(self) -> Dict:
        """Configuration for solar-powered deployment"""
        return {
            "power_profile": {
                "raspberry_pi_4_idle": 2.5,  # Watts
                "raspberry_pi_4_peak": 7.5,  # Watts
                "raspberry_pi_5_idle": 3.0,  # Watts
                "raspberry_pi_5_peak": 9.0,  # Watts
            },
            "solar_setup": {
                "panel_watts": 50,
                "battery_ah": 20,
                "battery_voltage": 12,
                "charge_controller": "PWM 10A",
                "estimated_runtime_hours": 18,
                "backup_days": 2
            },
            "power_optimization": {
                "cpu_governor": "ondemand",
                "disable_hdmi": True,
                "disable_bluetooth": True,
                "disable_wifi_when_idle": True,
                "screen_blank": 300,  # seconds
                "usb_auto_suspend": True
            }
        }

    def create_offline_sync_script(self) -> str:
        """Create script for periodic data sync when internet available"""
        script = """#!/bin/bash
# Guardião da Floresta - Offline Sync Script
# Syncs data when internet connection is available

SYNC_DIR="/home/pi/guardiao-da-floresta/data/sync"
REMOTE_SERVER="your-server.com"

echo "🔄 Checking internet connection..."

if ping -c 1 $REMOTE_SERVER &> /dev/null; then
    echo "🌐 Internet available. Syncing data..."

    # Sync alerts
    rsync -avz $SYNC_DIR/alerts/ user@$REMOTE_SERVER:/data/alerts/

    # Sync logs
    rsync -avz $SYNC_DIR/logs/ user@$REMOTE_SERVER:/data/logs/

    # Download model updates (if any)
    # rsync -avz user@$REMOTE_SERVER:/models/updates/ $SYNC_DIR/updates/

    echo "✅ Sync complete"
else
    echo "📴 No internet. Operating in offline mode."
    echo "📊 Data will be synced when connection is available."
fi
"""

        script_path = self.project_dir / "sync.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)

        # Create cron job for periodic sync
        cron_job = "0 */6 * * * /home/pi/guardiao-da-floresta/sync.sh >> /home/pi/guardiao-da-floresta/sync.log 2>&1"

        print(f"✅ Sync script created: {script_path}")
        print(f"📋 Add to crontab: {cron_job}")

        return str(script_path)

    def benchmark_device(self, device_type: str = "raspberry_pi_4") -> Dict:
        """Benchmark device performance"""
        import time
        import psutil

        print(f"🔬 Benchmarking {device_type}...")

        # CPU benchmark
        start = time.time()
        _ = sum(i**2 for i in range(10**6))
        cpu_time = time.time() - start

        # Memory benchmark
        mem = psutil.virtual_memory()

        # Estimate Gemma performance
        config = self.config["supported_devices"].get(device_type, {})
        expected_tps = config.get("expected_tokens_per_sec", 5)

        results = {
            "device": device_type,
            "cpu_benchmark_sec": cpu_time,
            "total_ram_gb": mem.total / (1024**3),
            "available_ram_gb": mem.available / (1024**3),
            "estimated_tokens_per_sec": expected_tps,
            "estimated_inference_time_100tokens": 100 / expected_tps,
            "power_consumption_watts": config.get("ram_gb", 4) * 1.5,
            "deployment_ready": mem.available > 3 * (1024**3)  # Need 3GB free
        }

        print(f"✅ Benchmark complete:")
        print(f"   CPU Time: {cpu_time:.2f}s")
        print(f"   RAM: {results['available_ram_gb']:.1f}GB available")
        print(f"   Estimated TPS: {expected_tps}")
        print(f"   Ready for deployment: {results['deployment_ready']}")

        return results

    def generate_deployment_package(self, device_type: str = "raspberry_pi_4") -> str:
        """Generate complete deployment package"""
        print(f"📦 Generating deployment package for {device_type}...")

        # Create package directory
        package_dir = self.project_dir / f"deploy-{device_type}"
        package_dir.mkdir(exist_ok=True)

        # Copy essential files
        files_to_copy = [
            "guardiao_core.py",
            "app.py",
            "requirements.txt",
            "Dockerfile",
            "data/"
        ]

        for file in files_to_copy:
            src = self.project_dir / file
            dst = package_dir / file
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.exists():
                shutil.copy2(src, dst)

        # Generate deployment scripts
        self.create_install_script(device_type)
        self.create_systemd_service(device_type)
        self.create_offline_sync_script()

        # Copy scripts to package
        for script in ["install.sh", "guardiao.service", "sync.sh"]:
            src = self.project_dir / script
            if src.exists():
                shutil.copy2(src, package_dir / script)

        # Create README for deployment
        deploy_readme = f"""# Guardião da Floresta - Deployment Package
## Device: {device_type}

## Quick Start
```bash
# 1. Copy to device (via USB or network)
scp -r deploy-{device_type} pi@raspberrypi.local:/home/pi/

# 2. Run installer
ssh pi@raspberrypi.local
cd /home/pi/deploy-{device_type}
./install.sh

# 3. Access application
# Open browser to http://raspberrypi.local:7860
```

## Manual Setup
See README.md in parent directory for detailed instructions.

## Support
For issues, contact: your-email@example.com
"""

        (package_dir / "DEPLOY_README.md").write_text(deploy_readme)

        # Create tarball
        tarball = shutil.make_archive(
            str(self.project_dir / f"guardiao-deploy-{device_type}"),
            "gztar",
            package_dir
        )

        print(f"✅ Deployment package created: {tarball}")
        print(f"   Size: {Path(tarball).stat().st_size / (1024**2):.1f} MB")

        return tarball

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🌿 GUARDIÃO DA FLORESTA - Edge Deployment Tool")
    print("=" * 60)

    deployer = EdgeDeployer()

    # Generate all deployment artifacts
    print("\n📦 Generating deployment artifacts...")

    # 1. Systemd service
    deployer.create_systemd_service()

    # 2. Install script
    deployer.create_install_script("raspberry_pi_4")

    # 3. Sync script
    deployer.create_offline_sync_script()

    # 4. Solar config
    solar_config = deployer.create_solar_power_config()
    with open("solar_config.json", "w") as f:
        json.dump(solar_config, f, indent=2)
    print("✅ Solar config saved: solar_config.json")

    # 5. Benchmark
    deployer.benchmark_device("raspberry_pi_4")

    # 6. Full deployment package
    # deployer.generate_deployment_package("raspberry_pi_4")

    print("\n" + "=" * 60)
    print("✅ All deployment artifacts generated!")
    print("=" * 60)
