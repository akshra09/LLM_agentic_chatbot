
from src.langgraphagenticai.state.state import State


class ChatbotWithToolNode:
    """
    Chatbot with tool node logic implementation
    """
    
    def __init__(self, model) -> None:
        self.llm = model
        
    def process(self, state:State) -> dict:
        """
        Process the input state and generate a chatbot response with tool integration
        """
        
        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])
        
        #simulate tool-specific logic
        tools_response = f"Tool integration for: '{user_input}'"
        
        return{"messages": [llm_response, tools_response]} 
    
    def create_chatbot(self, tools):
        '''
        Returns a chatbot node function
        '''
        
        llm_with_tools = self.llm.bind_tools(tools)
        
        def chatbot_node(state: State):
            '''
            Chatbot logic for processing the input state and returning a response
            '''
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        return chatbot_node
    
