from time import time


class Timer:
    """
    An object that keeps track of our progress in some repetitive loop and outputs a time estimate of the remaining time
    we will need to finish our loop. It is a handy tool for line and or grid searches, or sequential monte carlo
    simulations. Note that if the constituting steps in the loop(s) take vastly different times, the time estimate can
    be arbitrarily off, however the overall progress will still be displayed.
    """

    def __init__(self, total_steps):
        """
        Constructor.

        :param total_steps: (int) The total number of steps the measured process will make.
        """
        self.total_steps = total_steps
        self.total_time_elapsed = 0.
        self.recorded_steps = 0
        self.running_avg = None
        self.last_measured_time = None

        # time estimates
        self.remaining_seconds = None
        self.remaining_minutes = None
        self.remaining_hours = None

        # completion
        self.completion = 0.

    def __str__(self):
        self._calculate_completion_and_time_remaining()
        spaces = '          '  # to account for overhanging lines :)
        return f'{int(self.completion * 100)}%: {self.remaining_hours}h {self.remaining_minutes}m {self.remaining_seconds}s{spaces}'

    def start(self):
        """
        Mark the start of the innermost loop over which you wish to measure and record the current clock time.

        :return: None
        """
        self.last_measured_time = time()

    def end(self):
        """
        Mark the end of the innermost loop over which you wish to measure and record the current clock time. Eventually,
        update the running average estimate and add a step to the completed ones.

        :return: None
        """
        recorded_time = time() - self.last_measured_time
        self.total_time_elapsed += recorded_time
        if self.running_avg is None:
            self.running_avg = recorded_time
        else:
            self.running_avg = (self.running_avg * self.recorded_steps + recorded_time) / (self.recorded_steps + 1)
        self.recorded_steps += 1

    @staticmethod
    def _convert_seconds_to_h_m_s(seconds):
        """
        A private method to convert seconds into hours, minutes and seconds for better human readability.

        :param seconds: (int) Seconds we wish to convert into h, m, s format.
        :return: (tuple) The amount of seconds given converted into h, m, s format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds - hours * 3600) // 60)
        rem_seconds = int(seconds - hours * 3600 - minutes * 60)
        return hours, minutes, rem_seconds

    def _calculate_completion_and_time_remaining(self):
        """
        Private method to compute the completion of the process and estimate the remaining time from the running
        average.

        :return: None
        """
        remaining = self.total_steps - self.recorded_steps
        self.completion = self.recorded_steps / self.total_steps
        if self.running_avg is not None:
            estimated_time = remaining * self.running_avg
            self.remaining_hours, self.remaining_minutes, self.remaining_seconds = self._convert_seconds_to_h_m_s(estimated_time)
        else:
            self.remaining_hours = '??'
            self.remaining_minutes = '??'
            self.remaining_seconds = '??'

    def duration(self):
        """
        After the process has finished call this method to display the absolute time the completion of the whole process
        has taken.

        :return: None
        """
        h, m, s = self._convert_seconds_to_h_m_s(self.total_time_elapsed)
        print('\n')
        print(f'Completed. Time Elapsed: {h}h {m}m {s}s')
