"""O Tucano como analisador parcial (testemunha auditada).

O modelo não é tratado como um classificador cuja resposta deva ser
aceita, nem como um oráculo da ironia. Ele é empregado como um analisador
parcial, cujas leituras são submetidas a uma instância de decisão
discursiva explícita. A geração é determinística (do_sample=False),
condição da auditabilidade: a mesma entrada produz sempre a mesma saída.
"""

MODELO_PADRAO = "TucanoBR/Tucano-2b4-Instruct"


class TucanoAuditado:
    def __init__(self, nome_modelo=MODELO_PADRAO, device=None, max_new_tokens=220):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.nome_modelo = nome_modelo
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.modelo = AutoModelForCausalLM.from_pretrained(nome_modelo, torch_dtype=dtype)
        self.modelo.to(self.device)
        self.modelo.eval()

    def responder(self, prompt):
        """Geração determinística de uma resposta ao prompt de um eixo."""
        import torch

        mensagens = [{"role": "user", "content": prompt}]
        entrada = self.tokenizer.apply_chat_template(
            mensagens, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            saida = self.modelo.generate(
                entrada,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            saida[0][entrada.shape[-1]:], skip_special_tokens=True
        ).strip()
