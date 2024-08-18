from gradio_client import Client,file

client = Client("http://vlrlab-monkey.xyz:7681/")
result = client.predict(
		[["你是什么llm",'']],	# Tuple[str | Dict(file: filepath, alt_text: str | None) | None, str | Dict(file: filepath, alt_text: str | None) | None]  in 'Monkey-Chat' Chatbot component
		"Hello!",	# str  in 'Input' Textbox component
		api_name="/add_text"
)
print(result)
result = client.predict(
		[["Hello!",'']],	# Tuple[str | Dict(file: filepath, alt_text: str | None) | None, str | Dict(file: filepath, alt_text: str | None) | None]  in 'Monkey-Chat' Chatbot component
		api_name="/predict"
)
print(result)
result = client.predict(
		[["give caption!",'']],	# Tuple[str | Dict(file: filepath, alt_text: str | None) | None, str | Dict(file: filepath, alt_text: str | None) | None]  in 'Monkey-Chat' Chatbot component
		file("../img_captioning_CNNRNN/png/example.png"),
		api_name="/add_file"
)
print(result)
result = client.predict(
		[["Hello!",'']],	# Tuple[str | Dict(file: filepath, alt_text: str | None) | None, str | Dict(file: filepath, alt_text: str | None) | None]  in 'Monkey-Chat' Chatbot component
		api_name="/caption"
)
print(result)
