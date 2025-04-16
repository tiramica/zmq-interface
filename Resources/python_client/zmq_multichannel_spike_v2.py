import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import zmq
import json
import time

class SpikeViewer:
    def __init__(self, ip='localhost', port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.spikes = []  # List of (electrode, sorted_id, waveform)
        self.offset = 160  # Start where spikes are expected
        self.last_plot_time = time.time()
        self.max_channels = 384  # Update if needed based on your device

        self.init_gui()

    def init_gui(self):
        self.fig, self.axs = plt.subplots(4, 4, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2, hspace=0.6)

        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.offset_slider = Slider(ax_slider, 'Channel Offset', 0, self.max_channels - 16, valinit=self.offset, valstep=1)
        self.offset_slider.on_changed(self.update_offset)

    def update_offset(self, val):
        self.offset = int(val)
        self.plot_spikes()

    def plot_spikes(self):
        for ax in self.axs.flat:
            try:
                ax.clear()
                ax.axis('off')
            except Exception as e:
                print(f"Plot clear error: {e}")
                continue

        targets = list(range(self.offset, self.offset + 16))
        match_count = sum(1 for e, _, _ in self.spikes[-500:] if self.offset <= e < self.offset + 16)
        print(f"[Plot] Spikes in current view: {match_count}")

        for i, ch in enumerate(targets):
            if i >= len(self.axs.flat):
                break
            ax = self.axs.flat[i]
            waveforms = []

            for elec, _, wf in self.spikes[-500:]:
                if elec == ch:
                    for trace in wf:
                        ax.plot(trace, linewidth=0.5, alpha=0.3)
                    waveforms.append(wf)

            if waveforms:
                avg_waveform = np.mean([wf[0] for wf in waveforms], axis=0)
                ax.plot(avg_waveform, color='black', linewidth=1.5, label='Avg (ch0)')
                ax.set_title(f"Channel Electrode {ch}", fontsize=8)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_title(f"Channel Electrode {ch}\n(No spikes)", fontsize=8)
                ax.axis('on')
                ax.set_xticks([])
                ax.set_yticks([])

        self.fig.suptitle(f"Spike Overlays by Channels (offset {self.offset})", fontsize=12)
        self.fig.canvas.draw_idle()

    def run(self):
        print("🧠 Spike Viewer started. Waiting for spikes...")
        plt.ion()
        plt.show()
        while True:
            try:
                message = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                header = json.loads(message[1].decode('utf-8'))
                if header['type'] == 'spike':
                    content = header['spike']
                    electrode = content['electrode']
                    if isinstance(electrode, str) and 'Electrode' in electrode:
                        electrode = int(electrode.split()[-1])
                    sid = content['sorted_id']
                    num_channels = content['num_channels']
                    num_samples = content['num_samples']
                    data = np.frombuffer(message[2], dtype=np.float32)
                    waveform = np.reshape(data, (num_channels, num_samples))
                    self.spikes.append((electrode, sid, waveform))
                    print(f"Spike received: electrode={electrode}, shape={waveform.shape}")
                    if len(self.spikes) > 1000:
                        self.spikes.pop(0)

                if time.time() - self.last_plot_time > 0.2:
                    self.plot_spikes()
                    self.last_plot_time = time.time()

            except zmq.Again:
                plt.pause(0.01)

if __name__ == '__main__':
    viewer = SpikeViewer()
    viewer.run()
