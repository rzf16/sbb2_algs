from enum import Enum
from buffer import Buffer

class DMM:
    # Class for SBB Mealy machine

    class State(Enum):
        TERMINATE = 0
        ACTIVE = 1
        BUFFERING = 2
        WAITING = 3
    
    def __init__(self, major_buffer_max=600, wait_buffer_max=30,
                 pre_buffer_min=20, similarity_threshold=10, value_threshold=0):
        self.MAJOR_BUFFER_MAX = major_buffer_max
        self.WAIT_BUFFER_MAX = wait_buffer_max
        self.SIMILARITY_THRESHOLD = similarity_threshold
        self.VALUE_THRESHOLD = value_threshold
        self.PRE_BUFFER_MIN = pre_buffer_min
        self.started = False
    
    def start(self, precursor):
        self.state = DMM.State.ACTIVE
        self.prev_state = DMM.State.ACTIVE
        self.pre_buffer = precursor
        self.major_buffer = Buffer()
        self.wait_buffer = Buffer()
        self.started = True

        input = 2 if precursor.maxValue() > self.VALUE_THRESHOLD else 1
        print("DMM input: " + str(input))
        action = self.update_state(input)
        if action == 1 or action == 2:
            self.run_action(None, action)
        else:
            raise ValueError("Inappropriate initial action number " + str(action))
    
    def compute_input(self, next_frame, similarity=None):
        # Computes next input to DMM

        input = None

        if self.major_buffer.size() >= self.MAJOR_BUFFER_MAX:
            input = 5
        elif self.wait_buffer.size() >= self.WAIT_BUFFER_MAX:
            input = 6
        elif similarity is not None and similarity > self.SIMILARITY_THRESHOLD:
            if next_frame.value <= self.VALUE_THRESHOLD:
                input = 3
            else:
                input = 4
        else:
            if next_frame.value <= self.VALUE_THRESHOLD:
                input = 1
            else:
                input = 2
        
        print("DMM input: " + str(input))
        return input

    def update_state(self, input):
        # Updates machine state based on input
        # Returns action to take
        # Action 1: Active -> Buffering
        # Action 2: Active -> Waiting
        # Action 3: Buffering -> Buffering
        # Action 4: Waiting -> Buffering
        # Action 5: Buffering -> Waiting OR Waiting -> Waiting
        # Action 6: Buffering -> Terminate
        # Action 7: Waiting -> Terminate

        self.prev_state = self.state
        action = None

        if self.state == DMM.State.ACTIVE:
            if input == 1 or input == 3:
                self.state = DMM.State.WAITING
                action = 2
            elif input == 2 or input == 4:
                self.state = DMM.State.BUFFERING
                action = 1
            else:
                raise ValueError("Inappropriate DMM input " + str(input) +
                                 " to active state!")
        elif self.state == DMM.State.BUFFERING:
            if input == 1:
                self.state = DMM.State.WAITING
                action = 5
            elif input == 2 or input == 3 or input == 4:
                self.state = DMM.State.BUFFERING
                action = 3
            elif input == 5:
                self.state = DMM.State.TERMINATE
                action = 6
            else:
                raise ValueError("Inappropriate DMM input " + str(input) +
                                 " to buffering state!")
        elif self.state == DMM.State.WAITING:
            if input == 1 or input == 3:
                self.state = DMM.State.WAITING
                action = 5
            elif input == 2 or input == 4:
                self.state = DMM.State.BUFFERING
                action = 4
            elif input == 6:
                self.state = DMM.State.TERMINATE
                action = 7
            else:
                raise ValueError("Inappropriate DMM input " + str(input) +
                                 " to waiting state!")
        else:
            raise ValueError("Inappropriate DMM state " + DMM.State(self.state).name)
        
        print("DMM new state: " + DMM.State(self.state).name)
        print("DMM action number: " + str(action))
        return action

    def run_action(self, next_frame, action):
        # Runs the DMM action number specified by action
        # Returns True if the frame was resolved
        
        frame_resolved = True

        if action == 1:
            self.major_buffer.extend(self.pre_buffer)
            if next_frame is not None:
                self.major_buffer.append(next_frame)
            self.pre_buffer = Buffer()
        elif action == 2:
            self.wait_buffer.extend(self.pre_buffer)
            if next_frame is not None:
                self.wait_buffer.append(next_frame)
            self.pre_buffer = Buffer()
        elif action == 3:
            if next_frame is not None:
                self.major_buffer.append(next_frame)
        elif action == 4:
            self.major_buffer.extend(self.wait_buffer)
            if next_frame is not None:
                self.major_buffer.append(next_frame)
            self.wait_buffer = Buffer()
        elif action == 5:
            self.wait_buffer.append(next_frame)
        elif action == 6:
            self.pre_buffer.copy(self.major_buffer, copy_from=-self.PRE_BUFFER_MIN)
            self.major_buffer.split(split_until=-self.PRE_BUFFER_MIN)
            frame_resolved = False
        elif action == 7:
            pre_buffer_size = max(self.PRE_BUFFER_MIN,
                                  self.major_buffer.size() + self.wait_buffer.size() - self.MAJOR_BUFFER_MAX)
            self.pre_buffer.copy(self.wait_buffer, copy_from=-pre_buffer_size)
            self.major_buffer.extend(self.wait_buffer, extend_until=-pre_buffer_size)
            frame_resolved = False
        else:
            raise ValueError("Inappropriate DMM action number " + str(action))

        return frame_resolved

    def reset(self, next_frame):
        # Resets DMM after a terminate
        # Returns initial input given previous state
        # Terminate from BUFFERING -> 2
        # Terminate from WAITING -> 1

        input = None
        if self.prev_state == DMM.State.BUFFERING or next_frame.value > self.VALUE_THRESHOLD:
            input = 2
        elif self.prev_state == DMM.State.WAITING:
            input = 1
        else:
            raise ValueError("Inappropriate DMM state " + DMM.State(self.prev_state).name +
                             " during reset!")

        self.state = DMM.State.ACTIVE
        self.prev_state = DMM.State.ACTIVE
        self.major_buffer = Buffer()
        self.wait_buffer = Buffer()

        print("DMM input: " + str(input))
        return input