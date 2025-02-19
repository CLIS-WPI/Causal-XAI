import h5py
def save_performance_metrics(metrics, filepath):
    with h5py.File(filepath, 'a') as f:
        perf_group = f.create_group('performance_metrics')
        
        # Save beam switching metrics
        switch_group = perf_group.create_group('beam_switches')
        for idx, switch in enumerate(metrics['beam_switches']):
            switch_group.create_dataset(f'switch_{idx}', data=np.array([
                switch['timestamp'],
                switch['duration']
            ]))
            
        # Save BER history
        ber_group = perf_group.create_group('ber_history')
        ber_data = np.array([[b['timestamp'], b['value']] for b in metrics['ber_history']])
        ber_group.create_dataset('ber_data', data=ber_data)
        
        # Save SNR history
        snr_group = perf_group.create_group('snr_history')
        snr_data = np.array([[s['timestamp'], s['value']] for s in metrics['snr_history']])
        snr_group.create_dataset('snr_data', data=snr_data)
        
        # Save packet statistics
        packet_group = perf_group.create_group('packet_stats')
        packet_group.attrs['success_rate'] = metrics['packet_stats']['successful'] / metrics['packet_stats']['total']
        packet_group.attrs['switch_failure_rate'] = metrics['packet_stats']['failed_during_switch'] / metrics['packet_stats']['total']