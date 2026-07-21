/* Ogg-Opus encoding via WebCodecs AudioEncoder — no external libraries.
 *
 * Compresses the 16 kHz mono PCM to Opus at OPUS_BITRATE before upload
 * (roughly 5-6x smaller than 16-bit WAV). 48 kbit/s measured as the sweet
 * spot: transcripts differ ~2.7% from the WAV reference vs ~5.5% at 32k,
 * while 64k shows no further gain. The server decodes Ogg-Opus natively
 * (libsndfile, ffmpeg fallback). Callers must feature-check via
 * `oggOpusSupported()` and fall back to WAV when unsupported.
 */
"use strict";

/* CRC-32 as used by Ogg: poly 0x04C11DB7, not reflected, init 0, no xor. */
const OGG_CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let r = i << 24;
    for (let j = 0; j < 8; j++) {
      r = (r & 0x80000000) ? ((r << 1) ^ 0x04c11db7) : (r << 1);
    }
    t[i] = r >>> 0;
  }
  return t;
})();

function oggCrc(bytes) {
  let crc = 0;
  for (let i = 0; i < bytes.length; i++) {
    crc = (((crc << 8) >>> 0) ^ OGG_CRC_TABLE[((crc >>> 24) & 0xff) ^ bytes[i]]) >>> 0;
  }
  return crc >>> 0;
}

function oggPage(packets, { serial, pageSeq, granule, bos = false, eos = false }) {
  const lacing = [];
  for (const p of packets) {
    let n = p.length;
    while (n >= 255) { lacing.push(255); n -= 255; }
    lacing.push(n);
  }
  const bodyLen = packets.reduce((s, p) => s + p.length, 0);
  const page = new Uint8Array(27 + lacing.length + bodyLen);
  const v = new DataView(page.buffer);
  page.set([0x4f, 0x67, 0x67, 0x53, 0], 0); // "OggS", version 0
  page[5] = (bos ? 2 : 0) | (eos ? 4 : 0);
  // 64-bit little-endian granule position
  v.setUint32(6, granule >>> 0, true);
  v.setUint32(10, Math.floor(granule / 4294967296), true);
  v.setUint32(14, serial, true);
  v.setUint32(18, pageSeq, true);
  // CRC (22) filled below
  page[26] = lacing.length;
  page.set(lacing, 27);
  let off = 27 + lacing.length;
  for (const p of packets) { page.set(p, off); off += p.length; }
  v.setUint32(22, oggCrc(page), true);
  return page;
}

function defaultOpusHead(sampleRate) {
  const head = new Uint8Array(19);
  const v = new DataView(head.buffer);
  head.set([0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64]); // "OpusHead"
  head[8] = 1;   // version
  head[9] = 1;   // channels
  v.setUint16(10, 312, true);        // pre-skip @48k
  v.setUint32(12, sampleRate, true); // original input rate
  v.setUint16(16, 0, true);          // output gain
  head[18] = 0;  // mapping family
  return head;
}

function opusTags() {
  const vendor = new TextEncoder().encode("granite-speech-ui");
  const tags = new Uint8Array(8 + 4 + vendor.length + 4);
  const v = new DataView(tags.buffer);
  tags.set([0x4f, 0x70, 0x75, 0x73, 0x54, 0x61, 0x67, 0x73]); // "OpusTags"
  v.setUint32(8, vendor.length, true);
  tags.set(vendor, 12);
  v.setUint32(12 + vendor.length, 0, true); // no user comments
  return tags;
}

const OPUS_BITRATE = 48000;

async function oggOpusSupported(sampleRate) {
  if (typeof AudioEncoder === "undefined") return false;
  try {
    const { supported } = await AudioEncoder.isConfigSupported({
      codec: "opus", sampleRate, numberOfChannels: 1, bitrate: OPUS_BITRATE,
    });
    return !!supported;
  } catch {
    return false;
  }
}

/* samples: Float32Array (mono), sampleRate: e.g. 16000. Returns Blob (audio/ogg). */
async function encodeOggOpus(samples, sampleRate) {
  const packets = [];   // { data: Uint8Array, durationUs: number }
  let opusHead = null;
  let encodeError = null;

  const encoder = new AudioEncoder({
    output: (chunk, metadata) => {
      const desc = metadata?.decoderConfig?.description;
      if (desc && !opusHead) opusHead = new Uint8Array(desc.slice ? desc.slice(0) : desc);
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      packets.push({ data, durationUs: chunk.duration || 20000 });
    },
    error: (e) => { encodeError = e; },
  });
  encoder.configure({
    codec: "opus", sampleRate, numberOfChannels: 1, bitrate: OPUS_BITRATE,
  });

  const FRAME = Math.round(sampleRate / 50); // 20 ms per AudioData
  for (let pos = 0; pos < samples.length; pos += FRAME) {
    const slice = samples.subarray(pos, Math.min(pos + FRAME, samples.length));
    encoder.encode(new AudioData({
      format: "f32-planar",
      sampleRate,
      numberOfFrames: slice.length,
      numberOfChannels: 1,
      timestamp: Math.round(pos / sampleRate * 1e6),
      data: slice,
    }));
  }
  await encoder.flush();
  encoder.close();
  if (encodeError) throw encodeError;
  if (!packets.length) throw new Error("Opus-Encoder lieferte keine Daten");

  const head = opusHead || defaultOpusHead(sampleRate);
  const preSkip = new DataView(head.buffer, head.byteOffset).getUint16(10, true);

  const serial = Math.floor(Math.random() * 0xffffffff) >>> 0;
  const pages = [];
  let pageSeq = 0;
  pages.push(oggPage([head], { serial, pageSeq: pageSeq++, granule: 0, bos: true }));
  pages.push(oggPage([opusTags()], { serial, pageSeq: pageSeq++, granule: 0 }));

  const PACKETS_PER_PAGE = 50;
  let granule = preSkip;
  for (let i = 0; i < packets.length; i += PACKETS_PER_PAGE) {
    const batch = packets.slice(i, i + PACKETS_PER_PAGE);
    for (const p of batch) granule += Math.round(p.durationUs * 48000 / 1e6);
    pages.push(oggPage(batch.map((p) => p.data), {
      serial,
      pageSeq: pageSeq++,
      granule,
      eos: i + PACKETS_PER_PAGE >= packets.length,
    }));
  }
  return new Blob(pages, { type: "audio/ogg" });
}
