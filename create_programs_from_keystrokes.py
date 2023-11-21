import ast
class Program(object):
    def __init__(self, subject_id, assignment_id, file_name):
        self.subject_id = subject_id
        self.assignment_id = assignment_id
        self.file_name = file_name
        self.states = [""]
        self.asts = [ast.parse("")]
        self.executions = []
        
    def add_state(self, state):
        self.states.append(state)
        
        # If a program has no state progression, put the last valid AST
        try:
            self.asts.append(ast.parse(state))
        except:
            self.asts.append(self.asts[-1])
        
    def add_execution(self, metadata):
        if metadata != "Start":
            metadata = "End"
            
        execution = {
            "state_index": len(self.states),
            "execution": metadata,
        }
        self.executions.append(execution)
        
    # Remove comments first
    def clean_states(self):
        self.remove_comments()
        self.replace_tabs()
        self.condense_spacing()
        
    # Remove trailing comments and convert full-line comments to new-lines
    def remove_comments(self):
        for i in range(len(self.states)):
            self.states[i] = re.sub(r"\s*#.*?\n", r"\n", self.states[i])
    
    # Change 4-spaces into a tab character
    def replace_tabs(self):
        for i in range(len(self.states)):
            self.states[i] = re.sub(r"    ", r"\t", self.states[i])
            
    # Remove trailing whitespace and empty newlines
    def condense_spacing(self):
        for i in range(len(self.states)):
            self.states[i] = re.sub(r"\s*\n", r"\n", self.states[i])
        
    def __repr__(self):
        return (
            #f"Subject: {self.subject_id}\n" +
            #f"Assignment: {self.assignment_id}\n" +
            #f"File: {self.file_name}\n" +
            #f"States: {len(self.states)}\n" +
            #f"Executions: {len(self.executions)}\n\n" +
            #f"Final state...\n\n" +
            f"{self.states[-1]}"
        )


def build_program_string(subject_id, assignment_id, file_name, df):
    
    program = Program(subject_id, assignment_id, file_name)

    df = keystrokes[
        (keystrokes.SubjectID == subject_id) &
        (keystrokes.AssignmentID == assignment_id) &
        (keystrokes.CodeStateSection == file_name) &
        (keystrokes.EventType.isin(["File.Edit", "Run.Program"]))
    ].copy()

    df = df.fillna({"InsertText": "", "DeleteText": ""})

    state = ""
    for event_id, row in df.iterrows():
        if row.EventType == "Run.Program":
            program.add_execution(row["X-Metadata"])
            continue

        i = row.SourceLocation
        state = state[:i] + row.InsertText + state[i + len(row.DeleteText):]
        program.add_state(state)
        
    return program