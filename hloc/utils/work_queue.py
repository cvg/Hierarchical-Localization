from queue import Queue
from threading import Thread

class WorkQueue():
  def __init__(self, process_item, num_threads=4):

    self.queue = Queue(num_threads)
    self.threads = [Thread(target=self.write_thread, args=(process_item,)) 
      for _ in range(num_threads)]

    for thread in self.threads:
      thread.start()

  def join(self):

    for thread in self.threads:
      self.queue.put(None)

    for thread in self.threads:
      thread.join()


  def write_thread(self, process_item):
      item = self.queue.get()

      while item is not None:
        process_item(item)
        item = self.queue.get()

  def put(self, name, pred):
    self.queue.put( (name, pred) )
