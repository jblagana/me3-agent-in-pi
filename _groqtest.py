import json, io, wave, time, urllib.request, uuid, sys
import pyaudio

KEY = '***REMOVED-GROQ-KEY***'
GROQ = 'https://api.groq.com/openai/v1'
MODEL_ASR = 'whisper-large-v3-turbo'
MODEL_LLM = 'llama-3.3-70b-versatile'


def get_mic():
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d['name'].find('EMEET M1A') != -1:
            return i
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            return i
    return -1


def capture_wav(dev, secs=6):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input=True, input_device_index=dev, frames_per_buffer=1024)
    buf = io.BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
    print('CAPTURE: speak now (%ds)...' % secs, flush=True)
    t0 = time.time()
    while time.time() - t0 < secs:
        wf.writeframes(stream.read(1024, exception_on_overflow=False))
    stream.stop_stream(); stream.close(); pa.terminate(); wf.close()
    return buf.getvalue()


def groq_asr(wav):
    boundary = '----me3' + uuid.uuid4().hex
    b = bytearray()
    b += ('--' + boundary + '\r\n').encode()
    b += (b'Content-Disposition: form-data; name="file"; '
          b'filename="a.wav"\r\n')
    b += b'Content-Type: audio/wav\r\n\r\n'
    b += wav
    b += b'\r\n'
    b += ('--' + boundary + '\r\n').encode()
    b += b'Content-Disposition: form-data; name="model"\r\n\r\n'
    b += MODEL_ASR.encode()
    b += b'\r\n'
    req = urllib.request.Request(
        GROQ + '/audio/transcriptions', data=bytes(b),
        headers={'Authorization': 'Bearer ' + KEY,
                 'Content-Type': 'multipart/form-data; boundary=' + boundary})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get('text', '')


def groq_llm(prompt):
    body = {'model': MODEL_LLM, 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 80, 'temperature': 0.4}
    req = urllib.request.Request(
        GROQ + '/chat/completions', data=json.dumps(body).encode(),
        headers={'Authorization': 'Bearer ' + KEY,
                 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


if __name__ == '__main__':
    dev = get_mic()
    print('MIC DEV:', dev, flush=True)
    wav = capture_wav(dev)
    print('WAV bytes:', len(wav), flush=True)
    try:
        t0 = time.time(); txt = groq_asr(wav)
        print('ASR (%.2fs): %r' % (time.time() - t0, txt), flush=True)
    except Exception as e:
        print('ASR FAILED:', repr(e), flush=True); sys.exit(1)
    try:
        t0 = time.time()
        llm = groq_llm('User said: ' + (txt or '<empty>') +
                       '. Reply in 5 words or fewer.')
        print('LLM (%.2fs): %r' % (time.time() - t0, llm), flush=True)
    except Exception as e:
        print('LLM FAILED:', repr(e), flush=True); sys.exit(2)
    print('GROQ_TEST_OK', flush=True)
