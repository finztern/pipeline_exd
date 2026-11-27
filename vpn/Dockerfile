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
COPY healthcheck.sh /vpn/healthcheck.sh
COPY links.py subscription.py sbconfig.py manager.py build_config.py /vpn/
RUN chmod +x /entrypoint.sh /vpn/healthcheck.sh && mkdir -p /state

EXPOSE 1080 1081 9091

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=90s \
    CMD /vpn/healthcheck.sh || exit 1

ENTRYPOINT ["/entrypoint.sh"]
