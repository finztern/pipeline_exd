FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then A="arm64"; else A="amd64"; fi && \
    TAG=$(curl -fsSLI -o /dev/null -w '%{url_effective}' https://github.com/SagerNet/sing-box/releases/latest | sed 's#.*/tag/##') && \
    VER="${TAG#v}" && \
    curl -fsSL "https://github.com/SagerNet/sing-box/releases/download/${TAG}/sing-box-${VER}-linux-${A}.tar.gz" -o /tmp/sb.tgz && \
    tar -xzf /tmp/sb.tgz -C /tmp && \
    install -m 0755 /tmp/sing-box-${VER}-linux-${A}/sing-box /usr/local/bin/sing-box && \
    rm -rf /tmp/sb.tgz /tmp/sing-box-* && \
    sing-box version

WORKDIR /vpn
COPY entrypoint.sh /entrypoint.sh
COPY build_config.py /vpn/build_config.py
RUN chmod +x /entrypoint.sh

EXPOSE 1080 1081

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=10s \
    CMD curl -sf -x socks5h://127.0.0.1:1080 https://www.gstatic.com/generate_204 -o /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
