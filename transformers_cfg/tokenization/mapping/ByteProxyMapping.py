from transformers import AutoTokenizer


from typing import Dict, List


class ByteProxyMapping:
    def __init__(self, tokenizer):
        # check if the tokenizer is fast, if so, convert it to slow
        if tokenizer.is_fast:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer.name_or_path, use_fast=False
            )
        self.tokenizer = tokenizer

        # if tokenizer doesn't have byte_encoder(which is the case for llama-3), use gpt2_tokenizer
        if not hasattr(tokenizer, "byte_encoder"):
            gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
            tokenizer.byte_encoder = gpt2_tokenizer.byte_encoder
            tokenizer.byte_decoder = gpt2_tokenizer.byte_decoder

        self.byte2proxychar: Dict[int, str] = tokenizer.byte_encoder
        self.proxychar2byte: Dict[str, int] = tokenizer.byte_decoder

        # code point to byte
        self.cdp2byte: Dict[int, int] = {
            ord(c): b for c, b in self.proxychar2byte.items()
        }
        self.byte2cdp: Dict[int, int] = {v: k for k, v in self.cdp2byte.items()}
        self.PROXY_CDP_SET = set(self.cdp2byte.keys())
        # [33, 126] and [161,172, [174, 323], in total 94 + 12 + 150 = 256(N.B. 173 is a control character)

    def encode_byte2proxychar_cdp(self, byte: int) -> int:
        assert 0 <= byte < 256, f"byte: {byte} is not in the range [0, 256)"
        return ord(self.byte2proxychar[byte])

    def decode_proxychar2byte_cdp(self, cdp: int) -> int:
        byte_int: int = self.cdp2byte[cdp]
        assert 0 <= byte_int < 256, f"byte: {byte_int} is not in the range [0, 256)"
        return byte_int

    def decode_proxytoken2bytes(self, proxy_token: str) -> bytes:
        bytes_seq: List[int] = [
            self.decode_proxychar2byte_cdp(ord(c)) for c in proxy_token
        ]
        return bytes(bytes_seq)

    def map(self, proxy_token: str) -> bytes:
        return self.decode_proxytoken2bytes(proxy_token)

    def token2bytes(self, token: str) -> bytes:
        bytes_seq: List[int] = [self.proxychar2byte[c] for c in token]
        return bytes(bytes_seq)


class LLAMAByteProxyMapping:
    def __init__(self):
        pass

    def map(self, proxy_token: str) -> bytes:
        return self.decode_proxytoken2bytes(proxy_token)

    def decode_proxytoken2bytes(self, proxy_token: str) -> bytes:
        if proxy_token.startswith("<0x"):
            hex_value: str = proxy_token[3:-1]
            return bytes.fromhex(hex_value)
        else:
            # ad hoc fix for BPE
            if proxy_token.startswith("▁"):
                proxy_token = proxy_token.replace("▁", " ")
            return proxy_token.encode("utf-8")


if __name__ == "__main__":

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # gpt2_tokenizer.encode("´")

    byteproxymapper = ByteProxyMapping(gpt2_tokenizer)

    for i in range(256):
        print(f"{i}: {byteproxymapper.encode_byte2proxychar_cdp(i)}")

    # decode a byte

    byte = 162
    print(f"proxy code point set: {byteproxymapper.PROXY_CDP_SET}")
    print(f"len(proxy code point set): {len(byteproxymapper.PROXY_CDP_SET)}")
    for i in range(33, 127):
        if i not in byteproxymapper.PROXY_CDP_SET:
            print(f"{i} not in proxy code point set")
    for i in range(161, 324):
        if i not in byteproxymapper.PROXY_CDP_SET:
            print(f"{i} not in proxy code point set")
