import docker
import time

def start_producer(id, kafka_url, topic_name):
  print("> start producer !")
  docker_client = docker.from_env()
  container = docker_client.containers.run(
    "mmedhat1910/dew24_streaming_producer",
    detach=True,
    name=f"m2_producer_container_{int(time.time())}",
    environment={
      "ID": id,
      "KAFKA_URL":kafka_url,
      "TOPIC":topic_name,
      'debug': 'True'
    },
    network='de_ms2_met_p1_52_24625_default'
  )

  print('Container initialized:', container.id)
  return container.id

def stop_container(container_id):
  print('> Stop producer !')
  docker_client = docker.from_env()
  container = docker_client.containers.get(container_id)
  container.stop()
  container.remove()
  print('Container stopped:', container_id)
  return True