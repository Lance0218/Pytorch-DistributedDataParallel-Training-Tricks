#################################
# python
#################################
PYTHON := $$(which python3)
PIP := $(PYTHON) -m pip

.PHONY: install-all
install-all: install-other install-torch install-apex

.PHONY: install-other
install-other:
	@$(PIP) install -r requirements.txt

.PHONY: install-torch
install-torch:
	@$(PIP) install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

.PHONY: install-apex
install-apex:
	@git clone https://github.com/NVIDIA/apex ../apex
	@cd ../apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./