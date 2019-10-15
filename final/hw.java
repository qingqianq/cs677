import java.util.*;
import javax.security.auth.*;
class Treap<E extends Comparable<E>>{
	private Random priorityGenerator;
	private Node<E> root;
	private class Node<E>{
		public E data;
		public int priority;
		public Node<E> left;
		public Node<E> right;
		public Node<E> parent;
		public Node(E data, int priority){
			if (data == null)
				throw new RuntimeException("Data is null");
			this.data = data;
			this.priority = priority;
			this.left = null;
			this.right = null;
			this.parent = null;
		}
		Node<E> rotateLeft(){
			if(this == null || this.right == null){
				throw new RuntimeException("Try to rotate left when this.right node is null");
//				return this;
			}
			if(this == root){
				Node tmp = root.right.left;
				root.right.left = root;
				root = root.right;
				root.left.right = tmp;
				return this;
			}else {
				return this;
			}
		}
		Node<E> rotateRight(){
			if(this == null || this.left == null){
				throw new RuntimeException("Try to rotate right when this.left node is null");
//				return root;
			}
			Node<E> res = this;
			Node<E> tmp = res.left.right;
			res.left.right = res;
			res = res.left;
			res.right.left = tmp;
			return res;
		}
	}
	public Treap(){
		priorityGenerator = new Random();
		root = null;
	}
	public Treap(long seed){
		priorityGenerator = new Random(seed);
		root = null;
	}
	boolean add(E key){
		return add(key, priorityGenerator.nextInt());
	}
	boolean add(E key, int priority){
		if(root == null){
			root = new Node<E>(key, priority);
			return true;	
		}else {
			//TODO: vertify priority unique
			Stack<Node<E>> stack = new Stack<Node<E>>();
			stack.push(root);
			while (!stack.isEmpty()) {
				Node<E> tmp = stack.pop();
				if(tmp != null){
					if(tmp.priority == priority)
						return false;
					stack.push(tmp.right);
					stack.push(tmp.left);
				}
			}
			//TODO: add element without priority;
			Node<E> parent = null;
			Node<E> tmp = root;
			int compareResult = 0;
			while (tmp != null) {
				compareResult = key.compareTo(tmp.data);
				if(compareResult == 0)
					return false;
				parent = tmp;
				if(compareResult > 0)
					tmp = tmp.right;
				else
					tmp = tmp.left;
			}
			Node<E> nodeToInsert = new Node<E>(key, priority);
			nodeToInsert.parent = parent;
			if(compareResult > 0){
				parent.right = nodeToInsert;
//				if(parent.priority < nodeToInsert.priority){
//					parent = parent.rotateLeft();
//				}
			}else {
				parent.left = nodeToInsert;
//				if(parent.priority < nodeToInsert.priority){
//					parent = parent.rotateRight();
//				}
			}
			//TODO: rotate with priority
			return true;
		}	

	}
	boolean delete(E key){
		return true;
	}
	private boolean find(Node<E> root, E key){
		if(root == null)
			return false;
		Node<E> tmp = root;
		while (tmp != null) {
			if (tmp.data.compareTo(key) == 0) 
				return true;
			if(tmp.data.compareTo(key) > 0)
				tmp = tmp.left;
			else 
				tmp = tmp.right;
		}
		return false;
	}
	public boolean find(E key){
		return find(this.root, key);
	}
	public String toString(){
		if(root == null)
			return "null";
		StringBuilder sb = new StringBuilder();
		LinkedList<Node<E>> stack = new LinkedList<Node<E>>();
		stack.push(root);
		int n = 0;
		while (!stack.isEmpty()) {
//			int n = 0;
			Node<E> tmp = stack.pop();
			if (tmp != null) {
				sb.append("(key="+tmp.data+" , priority="+tmp.priority+")\n");
				stack.push(tmp.right);
				stack.push(tmp.left);
			}
			++n;
		}
		return sb.toString();
	}
	public static void main(String[] args) {
		Treap<Integer> t = new Treap<Integer>();
		t.add(4, 14);
//		t.add(2, 31);
//		t.add(6, 70);
//		t.add(1, 84);
//		t.add(3, 12);
//		t.add(5, 83);
//		t.add(7, 26);

	}
}