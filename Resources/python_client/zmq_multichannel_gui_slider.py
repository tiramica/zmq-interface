import zmq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import json
import time
class ZMQMultiChannelViewer:
    def __init__(self, ip='localhost', port=5556, num_total_channels=384, num_samples_display=1000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f'tcp://{ip}:{port}')
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.num_total_channels = num_total_channels
        self.num_samples_display = num_samples_display
        self.start_channel = 0
        self.window_size = 30

        self.buffer = np.zeros((num_total_channels, num_samples_display), dtype=np.float32)

        # Control flags
        self.paused = False
        self.running = True

        # Plot setup
        self.fig, self.axs = plt.subplots(5, 6, figsize=(15, 10))
        plt.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.25, hspace=0.6, wspace=0.4)

        self.init_gui()
        plt.ion()
        plt.show()

    def init_gui(self):
        # Slider for selecting start channel
        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        self.channel_slider = Slider(
            ax=ax_slider,
            label='Start Channel',
            valmin=0,
            valmax=self.num_total_channels - self.window_size,
            valinit=self.start_channel,
            valstep=1,
        )
        self.channel_slider.on_changed(self.update_channel_range)

        # Start button
        ax_start = plt.axes([0.2, 0.02, 0.1, 0.05])
        self.start_button = Button(ax_start, 'Start')
        self.start_button.on_clicked(self.start_plotting)

        # Pause button
        ax_pause = plt.axes([0.45, 0.02, 0.1, 0.05])
        self.pause_button = Button(ax_pause, 'Pause')
        self.pause_button.on_clicked(self.pause_plotting)

        # Quit button
        ax_quit = plt.axes([0.7, 0.02, 0.1, 0.05])
        self.quit_button = Button(ax_quit, 'Quit')
        self.quit_button.on_clicked(self.quit_viewer)

        self.update_plots()

    def update_channel_range(self, val):
        self.start_channel = int(self.channel_slider.val)
        print(f"✅ Updated channels: {self.start_channel}–{self.start_channel+self.window_size-1}")
        self.update_plots()

    def start_plotting(self, event):
        self.paused = False
        print("🟢 Plotting Resumed.")

    def pause_plotting(self, event):
        self.paused = True
        print("⏸️ Plotting Paused.")

    def quit_viewer(self, event):
        self.running = False
        print("⛔ Viewer Closing...")
        plt.close(self.fig)

    def update_buffer(self, channel, new_data):
        num_new_samples = len(new_data)
        if num_new_samples >= self.num_samples_display:
            self.buffer[channel, :] = new_data[-self.num_samples_display:]
        else:
            self.buffer[channel, :-num_new_samples] = self.buffer[channel, num_new_samples:]
            self.buffer[channel, -num_new_samples:] = new_data

    def update_plots(self):
        for ax in self.axs.flat:
            ax.clear()
            ax.axis('off')

        selected_channels = range(self.start_channel, self.start_channel + self.window_size)

        for i, ch in enumerate(selected_channels):
            row, col = divmod(i, 6)
            data = self.buffer[ch]
            ax = self.axs[row, col]
            ax.plot(data, linewidth=0.5)
            ax.set_title(f'Ch {ch}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('on')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def run(self):
        print("🟢 Neuropixels ZMQ Viewer Started.")
        messages_received = 0
        last_plot_update = 0
        total_samples_received = 0
        cycle_count = 0
        channel_counter = set()
        start_time = time.time()
        total_time = start_time
        channels_per_cycle=384
        channel_sampling_rates=[]
        # counter=0
        print(f"{'Elapsed (s)':>12} | {'Cycles/sec':>10} | {'Samples/sec':>15}| {'Avg.Sampling rates (Hz)':>10}")
        print('-'*44)

        while self.running:
            if self.paused:
                plt.pause(0.1)
                continue

            try:
                # message = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                message = self.socket.recv_multipart()
                header = json.loads(message[1].decode('utf-8'))
                
                if header['type'] == 'data':
                    content = header['content']
                    num_samples = content.get('num_samples', 0)
                    channel_num = content.get('channel_num', 0)
                    sampling_rate = content.get('sample_rate',0)
                    data = np.frombuffer(message[2], dtype=np.float32)
                    # print(data)
                    # print(f"Channel : {channel_num},numsamples{num_samples},data_length:{len(data)}")
                    total_samples_received += num_samples
                    channel_counter.add(channel_num)
                    channel_sampling_rates.append(sampling_rate)
                    # counter += 1
                    if len(channel_counter) == channels_per_cycle:
                        cycle_count += 1
                        channel_counter.clear()
                     
                    if len(data) != num_samples:
                        print(f"⚠️ Data mismatch Ch {channel_num}: Expected {num_samples}, got {len(data)}")
                        continue
                    # elapsed_ = time.time() - start_time
                    elapsed = time.time() - start_time
                    
                    if elapsed >= 0.0:
                        total_dev=time.time()-total_time
                        if total_dev>=1.0:
                            avg_sampling_rate = sum(channel_sampling_rates)/len(channel_sampling_rates)
                            print(f"{total_dev:12.2f} | {cycle_count:10d} | {total_samples_received:15d}| {avg_sampling_rate:10f}")
                            # reset for next interval
                            
                            # print(f"{elapsed:12.2f} | {cycle_count:10d} | {total_samples_received:15d}")
                            total_samples_received = 0
                            cycle_count = 0
                            counter = 0
                            total_time=time.time()
                        start_time = time.time()
                        
                    self.update_buffer(channel_num, data)
                    messages_received += 1

                    if messages_received - last_plot_update >= 384:
                        self.update_plots()
                        last_plot_update = messages_received
                    

            except zmq.Again:
                plt.pause(0.01)  # Prevent CPU overload when no data arrives

if __name__ == "__main__":
    viewer = ZMQMultiChannelViewer(ip='localhost', port=5556, num_total_channels=384)
    viewer.run()
